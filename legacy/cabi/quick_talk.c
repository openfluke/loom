/*
 * quick_talk.c  –  Interactive chat REPL using the Loom QuickTalk C-ABI
 *
 * All logic (history, system prompt, options, model path) lives here in C.
 * The Go/libloom side provides only handle-based primitives:
 *   QT_LoadTokenizer / QT_Encode / QT_Decode / QT_FreeTokenizer
 *   QT_LoadEngine    / QT_GetEngineInfo / QT_Generate / QT_FreeEngine
 *   QT_ReadEOSFromConfig
 *   FreeLoomString    (free any *C.char returned by the library)
 *
 * Compile example (Linux):
 *   gcc -I compiled/linux_x86_64 -o compiled/linux_x86_64/quick_talk \
 *       quick_talk.c -L compiled/linux_x86_64 -lloom \
 *       -Wl,-rpath,'$ORIGIN' -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include "libloom.h"

/* ── HuggingFace cache model discovery ─────────────────────────────────── */

#define MAX_MODELS 64

static char g_model_dirs[MAX_MODELS][2048]; /* resolved snapshot paths */
static char g_model_names[MAX_MODELS][256]; /* display names e.g. Qwen/Qwen2.5-0.5B */
static int  g_model_count = 0;

/* Replace the first occurrence of old_c with new_c in s (in-place). */
static void str_replace_first(char *s, char old_c, char new_c)
{
    char *p = strchr(s, old_c);
    if (p) *p = new_c;
}

/*
 * Scan <hub_dir> for entries whose name starts with "models--".
 * For each, navigate to snapshots/<first-entry>/ and store the path.
 * Populates g_model_dirs / g_model_names / g_model_count.
 */
static void discover_hf_models(const char *hub_dir)
{
    DIR *d = opendir(hub_dir);
    if (!d) return;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && g_model_count < MAX_MODELS) {
        if (strncmp(ent->d_name, "models--", 8) != 0) continue;

        /* Build snapshot path: hub_dir/models--Org--Model/snapshots */
        char snap_parent[2048];
        snprintf(snap_parent, sizeof(snap_parent),
                 "%s/%s/snapshots", hub_dir, ent->d_name);

        DIR *sd = opendir(snap_parent);
        if (!sd) continue;

        /* Pick the first snapshot hash directory */
        struct dirent *snap = NULL;
        struct dirent *se;
        while ((se = readdir(sd)) != NULL) {
            if (se->d_name[0] == '.') continue;
            snap = se;
            break;
        }
        if (!snap) { closedir(sd); continue; }

        snprintf(g_model_dirs[g_model_count], sizeof(g_model_dirs[0]),
                 "%s/%s", snap_parent, snap->d_name);
        closedir(sd);

        /* Turn "models--Org--Model" into "Org/Model" for display */
        char display[256];
        strncpy(display, ent->d_name + 8, sizeof(display) - 1); /* skip "models--" */
        display[sizeof(display)-1] = '\0';
        str_replace_first(display, '-', '/');  /* first '--' → '/' */
        /* collapse the remaining double-dash: "Org/--Model" → "Org/Model" */
        char *slash = strchr(display, '/');
        if (slash && slash[1] == '-') memmove(slash+1, slash+2, strlen(slash+2)+1);

        strncpy(g_model_names[g_model_count], display,
                sizeof(g_model_names[0]) - 1);
        g_model_count++;
    }
    closedir(d);
}

/*
 * Locate the HuggingFace hub directory and scan it.
 * Prints the numbered list; user selects. Returns 1 and fills model_dir,
 * or returns 0 if no models found / cache not found.
 */
static int pick_model_from_cache(char *model_dir, int model_dir_len)
{
    /* Determine home dir (Windows: USERPROFILE, Linux/macOS: HOME) */
    const char *home = getenv("USERPROFILE");
    if (!home || !home[0]) home = getenv("HOME");
    if (!home || !home[0]) return 0;

    char hub_dir[1024];
    snprintf(hub_dir, sizeof(hub_dir),
             "%s/.cache/huggingface/hub", home);

    discover_hf_models(hub_dir);

    if (g_model_count == 0) return 0;

    printf("Available models in HuggingFace cache:\n");
    for (int i = 0; i < g_model_count; i++)
        printf("  [%d] %s\n", i + 1, g_model_names[i]);

    char sel[64] = {0};
    printf("\nSelect model number [1]: ");
    fflush(stdout);
    if (!fgets(sel, sizeof(sel), stdin)) return 0;

    int idx = 1;
    sscanf(sel, "%d", &idx);
    if (idx < 1 || idx > g_model_count) idx = 1;

    strncpy(model_dir, g_model_dirs[idx - 1], model_dir_len - 1);
    model_dir[model_dir_len - 1] = '\0';
    return 1;
}

/* ── tiny JSON helpers ──────────────────────────────────────────────────── */

/* Extract a string field from a flat JSON object. Caller owns result. */
static char *json_get_str(const char *json, const char *key)
{
    if (!json || !key) return NULL;
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return NULL;
    p += strlen(needle);
    while (*p == ' ') p++;
    if (*p != '"') return NULL;
    p++;
    const char *end = strchr(p, '"');
    if (!end) return NULL;
    size_t len = (size_t)(end - p);
    char *out = (char *)malloc(len + 1);
    memcpy(out, p, len);
    out[len] = '\0';
    return out;
}

/* Extract an integer field from a flat JSON object, or return def. */
static long long json_get_int(const char *json, const char *key, long long def)
{
    if (!json || !key) return def;
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return def;
    p += strlen(needle);
    while (*p == ' ') p++;
    long long v = 0;
    if (sscanf(p, "%lld", &v) != 1) return def;
    return v;
}

/* ── history (dynamic array of turns) ──────────────────────────────────── */

typedef struct { char *user; char *assistant; } Turn;

static Turn   *g_history     = NULL;
static int     g_history_len = 0;
static int     g_history_cap = 0;

static void history_append(const char *user, const char *assistant)
{
    if (g_history_len >= g_history_cap) {
        g_history_cap = g_history_cap ? g_history_cap * 2 : 8;
        g_history = (Turn *)realloc(g_history, g_history_cap * sizeof(Turn));
    }
    g_history[g_history_len].user      = strdup(user);
    g_history[g_history_len].assistant = strdup(assistant);
    g_history_len++;
}

static void history_free(void)
{
    for (int i = 0; i < g_history_len; i++) {
        free(g_history[i].user);
        free(g_history[i].assistant);
    }
    free(g_history);
    g_history     = NULL;
    g_history_len = 0;
    g_history_cap = 0;
}

/* Serialise history to a JSON array string. Caller must free. */
static char *history_to_json(void)
{
    /* Rough upper bound: for each turn, user+assistant strings + JSON structure */
    size_t total = 3; /* "[]" + null */
    for (int i = 0; i < g_history_len; i++)
        total += strlen(g_history[i].user) * 2   + /* escape headroom */
                 strlen(g_history[i].assistant) * 2 + 64;

    char *buf = (char *)malloc(total);
    if (!buf) return strdup("[]");

    char *p = buf;
    *p++ = '[';
    for (int i = 0; i < g_history_len; i++) {
        if (i > 0) *p++ = ',';
        p += sprintf(p, "{\"user\":\"%s\",\"assistant\":\"%s\"}",
                     g_history[i].user, g_history[i].assistant);
    }
    *p++ = ']';
    *p   = '\0';
    return buf;
}

/* ── options builder ────────────────────────────────────────────────────── */

static char g_opts_json[1024];
static char g_sys_prompt[4096];
static int  g_max_tokens   = 50;
static float g_temperature  = 0.9f;
static int  g_top_k        = 40;
static int  g_kv_cache     = 1;
static float g_rep_penalty  = 1.15f;
static int  g_rep_window   = 64;
static int  g_deterministic = 0;
static char g_eos_json[256]; /* e.g. "[2,0]" */

static void build_opts_json(void)
{
    snprintf(g_opts_json, sizeof(g_opts_json),
             "{\"max_tokens\":%d,"
             "\"temperature\":%.3f,"
             "\"top_k\":%d,"
             "\"use_kv_cache\":%s,"
             "\"repetition_penalty\":%.3f,"
             "\"repetition_window\":%d,"
             "\"deterministic\":%s,"
             "\"eos_tokens\":%s}",
             g_max_tokens, g_temperature, g_top_k,
             g_kv_cache     ? "true" : "false",
             g_rep_penalty, g_rep_window,
             g_deterministic ? "true" : "false",
             g_eos_json[0] ? g_eos_json : "[2,0]");
}

/* ── input helpers ──────────────────────────────────────────────────────── */

static void trim_newline(char *s)
{
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r')) s[--n] = '\0';
}

static int str_eq_nocase(const char *a, const char *b)
{
    for (; *a && *b; a++, b++)
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
    return *a == '\0' && *b == '\0';
}

/* Read one line from stdin into buf (max len). Returns 0 on EOF. */
static int readline_into(char *buf, int len, const char *prompt)
{
    if (prompt) printf("%s", prompt);
    fflush(stdout);
    if (!fgets(buf, len, stdin)) return 0;
    trim_newline(buf);
    return 1;
}

/* Read a complete user message (single line or <<< multiline >>>). */
static int read_user_message(char *buf, int len)
{
    if (!readline_into(buf, len, "You: ")) return 0;
    if (str_eq_nocase(buf, "exit") || str_eq_nocase(buf, "quit")) return 0;

    if (strcmp(buf, "<<<") == 0) {
        printf("(paste mode – end with >>> on its own line)\n");
        buf[0] = '\0';
        char line[1024];
        while (fgets(line, sizeof(line), stdin)) {
            trim_newline(line);
            if (strcmp(line, ">>>") == 0) break;
            if (strlen(buf) + strlen(line) + 2 < (size_t)len) {
                if (buf[0]) strcat(buf, "\n");
                strcat(buf, line);
            }
        }
    }
    return buf[0] != '\0';
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(void)
{
    char model_dir[2048] = {0};
    char tok_path[2176]  = {0};
    char cfg_path[2176]  = {0};
    char line[1024]      = {0};

    printf("\n");
    printf("================================================\n");
    printf("  Loom QuickTalk - C REPL                      \n");
    printf("================================================\n");
    printf("\n");

    /* ── 1. Model directory ─────────────────────────────────────────────── */
    const char *env_path = getenv("LOOM_MODEL_PATH");
    if (env_path && env_path[0]) {
        strncpy(model_dir, env_path, sizeof(model_dir)-1);
        printf("[Model] Path (from LOOM_MODEL_PATH): %s\n", model_dir);
    } else if (!pick_model_from_cache(model_dir, sizeof(model_dir))) {
        /* Cache empty or not found – fall back to manual entry */
        printf("(No models found in HuggingFace cache)\n");
        readline_into(model_dir, sizeof(model_dir),
                      "[Model] Snapshot directory path: ");
    }
    printf("\n  Selected: %s\n", model_dir);


    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_dir);
    snprintf(cfg_path, sizeof(cfg_path), "%s/config.json",   model_dir);

    /* ── 2. Tokenizer ───────────────────────────────────────────────────── */
    printf("  Loading tokenizer...\n");
    long long tok_handle = QT_LoadTokenizer(tok_path);
    if (tok_handle < 0) {
        fprintf(stderr, "ERROR: failed to load tokenizer from %s\n", tok_path);
        return 1;
    }
    printf("  [OK] Tokenizer loaded (vocab: %d)\n", (int)QT_TokenizerVocabSize(tok_handle));

    /* ── 3. EOS tokens ──────────────────────────────────────────────────── */
    {
        char *eos_result = QT_ReadEOSFromConfig(cfg_path);
        if (eos_result) {
            /* Extract the eos_tokens array string from {"eos_tokens":[...]} */
            const char *arr_start = strstr(eos_result, ":[");
            if (arr_start) {
                arr_start++; /* skip ':' */
                const char *arr_end = strchr(arr_start, ']');
                if (arr_end) {
                    size_t arr_len = (size_t)(arr_end - arr_start) + 1;
                    if (arr_len < sizeof(g_eos_json)) {
                        memcpy(g_eos_json, arr_start, arr_len);
                        g_eos_json[arr_len] = '\0';
                    }
                }
            }
            printf("  [OK] EOS tokens: %s\n", g_eos_json);
            FreeLoomString(eos_result);
        }
    }

    /* ── 4. GPU choice ──────────────────────────────────────────────────── */
    int use_gpu = 0;
    readline_into(line, sizeof(line), "\n[GPU] Run on GPU? (1=yes / 0=no) [0]: ");
    use_gpu = (line[0] == '1') ? 1 : 0;

    /* ── 5. Max sequence length (GPU only) ──────────────────────────────── */
    int max_seq = 512;
    if (use_gpu) {
        readline_into(line, sizeof(line), "[Seq] Max sequence length [512]: ");
        if (line[0]) sscanf(line, "%d", &max_seq);
        if (max_seq <= 0) max_seq = 512;
    }

    /* ── 6. Template ────────────────────────────────────────────────────── */
    char tmpl[32] = "chatml";
    readline_into(line, sizeof(line), "[Tmpl] Template (chatml / llama3) [chatml]: ");
    if (line[0]) strncpy(tmpl, line, sizeof(tmpl)-1);

    /* ── 7. Load engine ─────────────────────────────────────────────────── */
    printf("  Loading model...\n");
    long long eng_handle = QT_LoadEngine(model_dir, use_gpu, max_seq, tmpl);
    if (eng_handle < 0) {
        fprintf(stderr, "ERROR: failed to load engine from %s\n", model_dir);
        QT_FreeTokenizer(tok_handle);
        return 1;
    }
    {
        char *info = QT_GetEngineInfo(eng_handle);
        if (info) {
            long long hidden = json_get_int(info, "hidden_size", 0);
            long long vocab  = json_get_int(info, "vocab_size",  0);
            long long layers = json_get_int(info, "num_layers",  0);
            char *gpu_str    = json_get_str(info, "gpu");
            printf("  [OK] Model loaded! hidden=%lld  vocab=%lld  layers=%lld  gpu=%s\n",
                   hidden, vocab, layers, gpu_str ? gpu_str : "false");
            free(gpu_str);
            FreeLoomString(info);
        }
    }

    /* ── 8. Generation options ──────────────────────────────────────────── */
    readline_into(line, sizeof(line), "\n[Mode] Deterministic? (1=yes / 0=no) [0]: ");
    g_deterministic = (line[0] == '1') ? 1 : 0;
    if (g_deterministic) { g_temperature = 0.0f; g_top_k = 1; }

    readline_into(line, sizeof(line), "[KV] KV-cache? (1=yes / 0=no) [1]: ");
    g_kv_cache = (line[0] == '0') ? 0 : 1;

    readline_into(line, sizeof(line), "[Tok] Max tokens per reply [50]: ");
    if (line[0]) sscanf(line, "%d", &g_max_tokens);
    if (g_max_tokens <= 0) g_max_tokens = 50;
    if (g_max_tokens > 512) { printf("  Clamping to 512.\n"); g_max_tokens = 512; }

    /* ── 9. Optional system prompt ──────────────────────────────────────── */
    printf("\n[Sys] System prompt (blank line to finish, empty = default):\n");
    g_sys_prompt[0] = '\0';
    char sp_line[512];
    while (fgets(sp_line, sizeof(sp_line), stdin)) {
        trim_newline(sp_line);
        if (sp_line[0] == '\0') break;
        if (strlen(g_sys_prompt) + strlen(sp_line) + 2 < sizeof(g_sys_prompt)) {
            if (g_sys_prompt[0]) strcat(g_sys_prompt, "\n");
            strcat(g_sys_prompt, sp_line);
        }
    }
    if (g_sys_prompt[0] == '\0')
        strncpy(g_sys_prompt,
                "You are a small, glitchy robot companion. "
                "Current Emotion: EXTREMELY HAPPY and EXCITED. "
                "You misunderstand insults as compliments. "
                "Be short, cute, and enthusiastic.",
                sizeof(g_sys_prompt)-1);

    build_opts_json();

    /* ── 10. Chat loop ──────────────────────────────────────────────────── */
    printf("\n[Chat] Type 'exit'/'quit' to stop, '<<<' for multiline, '!reset' to clear history.\n\n");

    char user_msg[4096];
    while (read_user_message(user_msg, sizeof(user_msg))) {

        if (str_eq_nocase(user_msg, "!reset")) {
            history_free();
            printf("(history cleared)\n\n");
            continue;
        }

        char *hist_json = history_to_json();
        char *result    = QT_Generate(eng_handle, tok_handle,
                                      hist_json, g_sys_prompt,
                                      user_msg, g_opts_json);
        free(hist_json);

        if (!result) { printf("(no response)\n\n"); continue; }

        char *reply = json_get_str(result, "reply");
        FreeLoomString(result);

        if (!reply) { printf("(empty reply)\n\n"); continue; }

        printf("Bot: %s\n\n", reply);

        history_append(user_msg, reply);
        free(reply);
    }

    printf("Goodbye!\n");

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    history_free();
    QT_FreeEngine(eng_handle);
    QT_FreeTokenizer(tok_handle);
    return 0;
}
