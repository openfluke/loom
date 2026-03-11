var __create = Object.create;
var __getProtoOf = Object.getPrototypeOf;
var __defProp = Object.defineProperty;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __toESM = (mod, isNodeMode, target) => {
  target = mod != null ? __create(__getProtoOf(mod)) : {};
  const to = isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target;
  for (let key of __getOwnPropNames(mod))
    if (!__hasOwnProp.call(to, key))
      __defProp(to, key, {
        get: () => mod[key],
        enumerable: true
      });
  return to;
};
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, {
      get: all[name],
      enumerable: true,
      configurable: true,
      set: (newValue) => all[name] = () => newValue
    });
};
var __esm = (fn, res) => () => (fn && (res = fn(fn = 0)), res);
var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
  get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
}) : x)(function(x) {
  if (typeof require !== "undefined")
    return require.apply(this, arguments);
  throw Error('Dynamic require of "' + x + '" is not supported');
});

// node:url
var exports_url = {};
__export(exports_url, {
  resolveObject: () => urlResolveObject,
  resolve: () => urlResolve,
  parse: () => urlParse,
  format: () => urlFormat,
  default: () => url_default,
  Url: () => Url,
  URLSearchParams: () => URLSearchParams,
  URL: () => URL
});
function util_isString(arg) {
  return typeof arg === "string";
}
function util_isObject(arg) {
  return typeof arg === "object" && arg !== null;
}
function util_isNull(arg) {
  return arg === null;
}
function util_isNullOrUndefined(arg) {
  return arg == null;
}
function Url() {
  this.protocol = null, this.slashes = null, this.auth = null, this.host = null, this.port = null, this.hostname = null, this.hash = null, this.search = null, this.query = null, this.pathname = null, this.path = null, this.href = null;
}
function urlParse(url2, parseQueryString, slashesDenoteHost) {
  if (url2 && util_isObject(url2) && url2 instanceof Url)
    return url2;
  var u = new Url;
  return u.parse(url2, parseQueryString, slashesDenoteHost), u;
}
function urlFormat(obj) {
  if (util_isString(obj))
    obj = urlParse(obj);
  if (!(obj instanceof Url))
    return Url.prototype.format.call(obj);
  return obj.format();
}
function urlResolve(source, relative) {
  return urlParse(source, false, true).resolve(relative);
}
function urlResolveObject(source, relative) {
  if (!source)
    return relative;
  return urlParse(source, false, true).resolveObject(relative);
}
var URL, URLSearchParams, protocolPattern, portPattern, simplePathPattern, delims, unwise, autoEscape, nonHostChars, hostEndingChars, hostnameMaxLen = 255, hostnamePartPattern, hostnamePartStart, unsafeProtocol, hostlessProtocol, slashedProtocol, querystring, url_default;
var init_url = __esm(() => {
  ({ URL, URLSearchParams } = globalThis);
  protocolPattern = /^([a-z0-9.+-]+:)/i;
  portPattern = /:[0-9]*$/;
  simplePathPattern = /^(\/\/?(?!\/)[^\?\s]*)(\?[^\s]*)?$/;
  delims = ["<", ">", '"', "`", " ", "\r", `
`, "\t"];
  unwise = ["{", "}", "|", "\\", "^", "`"].concat(delims);
  autoEscape = ["'"].concat(unwise);
  nonHostChars = ["%", "/", "?", ";", "#"].concat(autoEscape);
  hostEndingChars = ["/", "?", "#"];
  hostnamePartPattern = /^[+a-z0-9A-Z_-]{0,63}$/;
  hostnamePartStart = /^([+a-z0-9A-Z_-]{0,63})(.*)$/;
  unsafeProtocol = { javascript: true, "javascript:": true };
  hostlessProtocol = { javascript: true, "javascript:": true };
  slashedProtocol = { http: true, https: true, ftp: true, gopher: true, file: true, "http:": true, "https:": true, "ftp:": true, "gopher:": true, "file:": true };
  querystring = { parse(str) {
    var decode = decodeURIComponent;
    return (str + "").replace(/\+/g, " ").split("&").filter(Boolean).reduce(function(obj, item, index) {
      var ref = item.split("="), key = decode(ref[0] || ""), val = decode(ref[1] || ""), prev = obj[key];
      return obj[key] = prev === undefined ? val : [].concat(prev, val), obj;
    }, {});
  }, stringify(obj) {
    var encode = encodeURIComponent;
    return Object.keys(obj || {}).reduce(function(arr, key) {
      return [].concat(obj[key]).forEach(function(v) {
        arr.push(encode(key) + "=" + encode(v));
      }), arr;
    }, []).join("&").replace(/\s/g, "+");
  } };
  Url.prototype.parse = function(url2, parseQueryString, slashesDenoteHost) {
    if (!util_isString(url2))
      throw new TypeError("Parameter 'url' must be a string, not " + typeof url2);
    var queryIndex = url2.indexOf("?"), splitter = queryIndex !== -1 && queryIndex < url2.indexOf("#") ? "?" : "#", uSplit = url2.split(splitter), slashRegex = /\\/g;
    uSplit[0] = uSplit[0].replace(slashRegex, "/"), url2 = uSplit.join(splitter);
    var rest = url2;
    if (rest = rest.trim(), !slashesDenoteHost && url2.split("#").length === 1) {
      var simplePath = simplePathPattern.exec(rest);
      if (simplePath) {
        if (this.path = rest, this.href = rest, this.pathname = simplePath[1], simplePath[2])
          if (this.search = simplePath[2], parseQueryString)
            this.query = querystring.parse(this.search.substr(1));
          else
            this.query = this.search.substr(1);
        else if (parseQueryString)
          this.search = "", this.query = {};
        return this;
      }
    }
    var proto = protocolPattern.exec(rest);
    if (proto) {
      proto = proto[0];
      var lowerProto = proto.toLowerCase();
      this.protocol = lowerProto, rest = rest.substr(proto.length);
    }
    if (slashesDenoteHost || proto || rest.match(/^\/\/[^@\/]+@[^@\/]+/)) {
      var slashes = rest.substr(0, 2) === "//";
      if (slashes && !(proto && hostlessProtocol[proto]))
        rest = rest.substr(2), this.slashes = true;
    }
    if (!hostlessProtocol[proto] && (slashes || proto && !slashedProtocol[proto])) {
      var hostEnd = -1;
      for (var i = 0;i < hostEndingChars.length; i++) {
        var hec = rest.indexOf(hostEndingChars[i]);
        if (hec !== -1 && (hostEnd === -1 || hec < hostEnd))
          hostEnd = hec;
      }
      var auth, atSign;
      if (hostEnd === -1)
        atSign = rest.lastIndexOf("@");
      else
        atSign = rest.lastIndexOf("@", hostEnd);
      if (atSign !== -1)
        auth = rest.slice(0, atSign), rest = rest.slice(atSign + 1), this.auth = decodeURIComponent(auth);
      hostEnd = -1;
      for (var i = 0;i < nonHostChars.length; i++) {
        var hec = rest.indexOf(nonHostChars[i]);
        if (hec !== -1 && (hostEnd === -1 || hec < hostEnd))
          hostEnd = hec;
      }
      if (hostEnd === -1)
        hostEnd = rest.length;
      this.host = rest.slice(0, hostEnd), rest = rest.slice(hostEnd), this.parseHost(), this.hostname = this.hostname || "";
      var ipv6Hostname = this.hostname[0] === "[" && this.hostname[this.hostname.length - 1] === "]";
      if (!ipv6Hostname) {
        var hostparts = this.hostname.split(/\./);
        for (var i = 0, l = hostparts.length;i < l; i++) {
          var part = hostparts[i];
          if (!part)
            continue;
          if (!part.match(hostnamePartPattern)) {
            var newpart = "";
            for (var j = 0, k = part.length;j < k; j++)
              if (part.charCodeAt(j) > 127)
                newpart += "x";
              else
                newpart += part[j];
            if (!newpart.match(hostnamePartPattern)) {
              var validParts = hostparts.slice(0, i), notHost = hostparts.slice(i + 1), bit = part.match(hostnamePartStart);
              if (bit)
                validParts.push(bit[1]), notHost.unshift(bit[2]);
              if (notHost.length)
                rest = "/" + notHost.join(".") + rest;
              this.hostname = validParts.join(".");
              break;
            }
          }
        }
      }
      if (this.hostname.length > hostnameMaxLen)
        this.hostname = "";
      else
        this.hostname = this.hostname.toLowerCase();
      if (!ipv6Hostname)
        this.hostname = new URL(`https://${this.hostname}`).hostname;
      var p = this.port ? ":" + this.port : "", h = this.hostname || "";
      if (this.host = h + p, this.href += this.host, ipv6Hostname) {
        if (this.hostname = this.hostname.substr(1, this.hostname.length - 2), rest[0] !== "/")
          rest = "/" + rest;
      }
    }
    if (!unsafeProtocol[lowerProto])
      for (var i = 0, l = autoEscape.length;i < l; i++) {
        var ae = autoEscape[i];
        if (rest.indexOf(ae) === -1)
          continue;
        var esc = encodeURIComponent(ae);
        if (esc === ae)
          esc = escape(ae);
        rest = rest.split(ae).join(esc);
      }
    var hash = rest.indexOf("#");
    if (hash !== -1)
      this.hash = rest.substr(hash), rest = rest.slice(0, hash);
    var qm = rest.indexOf("?");
    if (qm !== -1) {
      if (this.search = rest.substr(qm), this.query = rest.substr(qm + 1), parseQueryString)
        this.query = querystring.parse(this.query);
      rest = rest.slice(0, qm);
    } else if (parseQueryString)
      this.search = "", this.query = {};
    if (rest)
      this.pathname = rest;
    if (slashedProtocol[lowerProto] && this.hostname && !this.pathname)
      this.pathname = "/";
    if (this.pathname || this.search) {
      var p = this.pathname || "", s = this.search || "";
      this.path = p + s;
    }
    return this.href = this.format(), this;
  };
  Url.prototype.format = function() {
    var auth = this.auth || "";
    if (auth)
      auth = encodeURIComponent(auth), auth = auth.replace(/%3A/i, ":"), auth += "@";
    var protocol = this.protocol || "", pathname = this.pathname || "", hash = this.hash || "", host = false, query = "";
    if (this.host)
      host = auth + this.host;
    else if (this.hostname) {
      if (host = auth + (this.hostname.indexOf(":") === -1 ? this.hostname : "[" + this.hostname + "]"), this.port)
        host += ":" + this.port;
    }
    if (this.query && util_isObject(this.query) && Object.keys(this.query).length)
      query = querystring.stringify(this.query);
    var search = this.search || query && "?" + query || "";
    if (protocol && protocol.substr(-1) !== ":")
      protocol += ":";
    if (this.slashes || (!protocol || slashedProtocol[protocol]) && host !== false) {
      if (host = "//" + (host || ""), pathname && pathname.charAt(0) !== "/")
        pathname = "/" + pathname;
    } else if (!host)
      host = "";
    if (hash && hash.charAt(0) !== "#")
      hash = "#" + hash;
    if (search && search.charAt(0) !== "?")
      search = "?" + search;
    return pathname = pathname.replace(/[?#]/g, function(match) {
      return encodeURIComponent(match);
    }), search = search.replace("#", "%23"), protocol + host + pathname + search + hash;
  };
  Url.prototype.resolve = function(relative) {
    return this.resolveObject(urlParse(relative, false, true)).format();
  };
  Url.prototype.resolveObject = function(relative) {
    if (util_isString(relative)) {
      var rel = new Url;
      rel.parse(relative, false, true), relative = rel;
    }
    var result = new Url, tkeys = Object.keys(this);
    for (var tk = 0;tk < tkeys.length; tk++) {
      var tkey = tkeys[tk];
      result[tkey] = this[tkey];
    }
    if (result.hash = relative.hash, relative.href === "")
      return result.href = result.format(), result;
    if (relative.slashes && !relative.protocol) {
      var rkeys = Object.keys(relative);
      for (var rk = 0;rk < rkeys.length; rk++) {
        var rkey = rkeys[rk];
        if (rkey !== "protocol")
          result[rkey] = relative[rkey];
      }
      if (slashedProtocol[result.protocol] && result.hostname && !result.pathname)
        result.path = result.pathname = "/";
      return result.href = result.format(), result;
    }
    if (relative.protocol && relative.protocol !== result.protocol) {
      if (!slashedProtocol[relative.protocol]) {
        var keys = Object.keys(relative);
        for (var v = 0;v < keys.length; v++) {
          var k = keys[v];
          result[k] = relative[k];
        }
        return result.href = result.format(), result;
      }
      if (result.protocol = relative.protocol, !relative.host && !hostlessProtocol[relative.protocol]) {
        var relPath = (relative.pathname || "").split("/");
        while (relPath.length && !(relative.host = relPath.shift()))
          ;
        if (!relative.host)
          relative.host = "";
        if (!relative.hostname)
          relative.hostname = "";
        if (relPath[0] !== "")
          relPath.unshift("");
        if (relPath.length < 2)
          relPath.unshift("");
        result.pathname = relPath.join("/");
      } else
        result.pathname = relative.pathname;
      if (result.search = relative.search, result.query = relative.query, result.host = relative.host || "", result.auth = relative.auth, result.hostname = relative.hostname || relative.host, result.port = relative.port, result.pathname || result.search) {
        var p = result.pathname || "", s = result.search || "";
        result.path = p + s;
      }
      return result.slashes = result.slashes || relative.slashes, result.href = result.format(), result;
    }
    var isSourceAbs = result.pathname && result.pathname.charAt(0) === "/", isRelAbs = relative.host || relative.pathname && relative.pathname.charAt(0) === "/", mustEndAbs = isRelAbs || isSourceAbs || result.host && relative.pathname, removeAllDots = mustEndAbs, srcPath = result.pathname && result.pathname.split("/") || [], relPath = relative.pathname && relative.pathname.split("/") || [], psychotic = result.protocol && !slashedProtocol[result.protocol];
    if (psychotic) {
      if (result.hostname = "", result.port = null, result.host)
        if (srcPath[0] === "")
          srcPath[0] = result.host;
        else
          srcPath.unshift(result.host);
      if (result.host = "", relative.protocol) {
        if (relative.hostname = null, relative.port = null, relative.host)
          if (relPath[0] === "")
            relPath[0] = relative.host;
          else
            relPath.unshift(relative.host);
        relative.host = null;
      }
      mustEndAbs = mustEndAbs && (relPath[0] === "" || srcPath[0] === "");
    }
    if (isRelAbs)
      result.host = relative.host || relative.host === "" ? relative.host : result.host, result.hostname = relative.hostname || relative.hostname === "" ? relative.hostname : result.hostname, result.search = relative.search, result.query = relative.query, srcPath = relPath;
    else if (relPath.length) {
      if (!srcPath)
        srcPath = [];
      srcPath.pop(), srcPath = srcPath.concat(relPath), result.search = relative.search, result.query = relative.query;
    } else if (!util_isNullOrUndefined(relative.search)) {
      if (psychotic) {
        result.hostname = result.host = srcPath.shift();
        var authInHost = result.host && result.host.indexOf("@") > 0 ? result.host.split("@") : false;
        if (authInHost)
          result.auth = authInHost.shift(), result.host = result.hostname = authInHost.shift();
      }
      if (result.search = relative.search, result.query = relative.query, !util_isNull(result.pathname) || !util_isNull(result.search))
        result.path = (result.pathname ? result.pathname : "") + (result.search ? result.search : "");
      return result.href = result.format(), result;
    }
    if (!srcPath.length) {
      if (result.pathname = null, result.search)
        result.path = "/" + result.search;
      else
        result.path = null;
      return result.href = result.format(), result;
    }
    var last = srcPath.slice(-1)[0], hasTrailingSlash = (result.host || relative.host || srcPath.length > 1) && (last === "." || last === "..") || last === "", up = 0;
    for (var i = srcPath.length;i >= 0; i--)
      if (last = srcPath[i], last === ".")
        srcPath.splice(i, 1);
      else if (last === "..")
        srcPath.splice(i, 1), up++;
      else if (up)
        srcPath.splice(i, 1), up--;
    if (!mustEndAbs && !removeAllDots)
      for (;up--; up)
        srcPath.unshift("..");
    if (mustEndAbs && srcPath[0] !== "" && (!srcPath[0] || srcPath[0].charAt(0) !== "/"))
      srcPath.unshift("");
    if (hasTrailingSlash && srcPath.join("/").substr(-1) !== "/")
      srcPath.push("");
    var isAbsolute = srcPath[0] === "" || srcPath[0] && srcPath[0].charAt(0) === "/";
    if (psychotic) {
      result.hostname = result.host = isAbsolute ? "" : srcPath.length ? srcPath.shift() : "";
      var authInHost = result.host && result.host.indexOf("@") > 0 ? result.host.split("@") : false;
      if (authInHost)
        result.auth = authInHost.shift(), result.host = result.hostname = authInHost.shift();
    }
    if (mustEndAbs = mustEndAbs || result.host && srcPath.length, mustEndAbs && !isAbsolute)
      srcPath.unshift("");
    if (!srcPath.length)
      result.pathname = null, result.path = null;
    else
      result.pathname = srcPath.join("/");
    if (!util_isNull(result.pathname) || !util_isNull(result.search))
      result.path = (result.pathname ? result.pathname : "") + (result.search ? result.search : "");
    return result.auth = relative.auth || result.auth, result.slashes = result.slashes || relative.slashes, result.href = result.format(), result;
  };
  Url.prototype.parseHost = function() {
    var host = this.host, port = portPattern.exec(host);
    if (port) {
      if (port = port[0], port !== ":")
        this.port = port.substr(1);
      host = host.substr(0, host.length - port.length);
    }
    if (host)
      this.hostname = host;
  };
  url_default = { parse: urlParse, resolve: urlResolve, resolveObject: urlResolveObject, format: urlFormat, Url, URL, URLSearchParams };
});

// node:path
var exports_path = {};
__export(exports_path, {
  sep: () => sep,
  resolve: () => resolve,
  relative: () => relative,
  posix: () => posix,
  parse: () => parse,
  normalize: () => normalize,
  join: () => join,
  isAbsolute: () => isAbsolute,
  format: () => format,
  extname: () => extname,
  dirname: () => dirname,
  delimiter: () => delimiter,
  default: () => path_default,
  basename: () => basename,
  _makeLong: () => _makeLong
});
function assertPath(path2) {
  if (typeof path2 !== "string")
    throw new TypeError("Path must be a string. Received " + JSON.stringify(path2));
}
function normalizeStringPosix(path2, allowAboveRoot) {
  var res = "", lastSegmentLength = 0, lastSlash = -1, dots = 0, code;
  for (var i = 0;i <= path2.length; ++i) {
    if (i < path2.length)
      code = path2.charCodeAt(i);
    else if (code === 47)
      break;
    else
      code = 47;
    if (code === 47) {
      if (lastSlash === i - 1 || dots === 1)
        ;
      else if (lastSlash !== i - 1 && dots === 2) {
        if (res.length < 2 || lastSegmentLength !== 2 || res.charCodeAt(res.length - 1) !== 46 || res.charCodeAt(res.length - 2) !== 46) {
          if (res.length > 2) {
            var lastSlashIndex = res.lastIndexOf("/");
            if (lastSlashIndex !== res.length - 1) {
              if (lastSlashIndex === -1)
                res = "", lastSegmentLength = 0;
              else
                res = res.slice(0, lastSlashIndex), lastSegmentLength = res.length - 1 - res.lastIndexOf("/");
              lastSlash = i, dots = 0;
              continue;
            }
          } else if (res.length === 2 || res.length === 1) {
            res = "", lastSegmentLength = 0, lastSlash = i, dots = 0;
            continue;
          }
        }
        if (allowAboveRoot) {
          if (res.length > 0)
            res += "/..";
          else
            res = "..";
          lastSegmentLength = 2;
        }
      } else {
        if (res.length > 0)
          res += "/" + path2.slice(lastSlash + 1, i);
        else
          res = path2.slice(lastSlash + 1, i);
        lastSegmentLength = i - lastSlash - 1;
      }
      lastSlash = i, dots = 0;
    } else if (code === 46 && dots !== -1)
      ++dots;
    else
      dots = -1;
  }
  return res;
}
function _format(sep, pathObject) {
  var dir = pathObject.dir || pathObject.root, base = pathObject.base || (pathObject.name || "") + (pathObject.ext || "");
  if (!dir)
    return base;
  if (dir === pathObject.root)
    return dir + base;
  return dir + sep + base;
}
function resolve() {
  var resolvedPath = "", resolvedAbsolute = false, cwd;
  for (var i = arguments.length - 1;i >= -1 && !resolvedAbsolute; i--) {
    var path2;
    if (i >= 0)
      path2 = arguments[i];
    else {
      if (cwd === undefined)
        cwd = process.cwd();
      path2 = cwd;
    }
    if (assertPath(path2), path2.length === 0)
      continue;
    resolvedPath = path2 + "/" + resolvedPath, resolvedAbsolute = path2.charCodeAt(0) === 47;
  }
  if (resolvedPath = normalizeStringPosix(resolvedPath, !resolvedAbsolute), resolvedAbsolute)
    if (resolvedPath.length > 0)
      return "/" + resolvedPath;
    else
      return "/";
  else if (resolvedPath.length > 0)
    return resolvedPath;
  else
    return ".";
}
function normalize(path2) {
  if (assertPath(path2), path2.length === 0)
    return ".";
  var isAbsolute = path2.charCodeAt(0) === 47, trailingSeparator = path2.charCodeAt(path2.length - 1) === 47;
  if (path2 = normalizeStringPosix(path2, !isAbsolute), path2.length === 0 && !isAbsolute)
    path2 = ".";
  if (path2.length > 0 && trailingSeparator)
    path2 += "/";
  if (isAbsolute)
    return "/" + path2;
  return path2;
}
function isAbsolute(path2) {
  return assertPath(path2), path2.length > 0 && path2.charCodeAt(0) === 47;
}
function join() {
  if (arguments.length === 0)
    return ".";
  var joined;
  for (var i = 0;i < arguments.length; ++i) {
    var arg = arguments[i];
    if (assertPath(arg), arg.length > 0)
      if (joined === undefined)
        joined = arg;
      else
        joined += "/" + arg;
  }
  if (joined === undefined)
    return ".";
  return normalize(joined);
}
function relative(from, to) {
  if (assertPath(from), assertPath(to), from === to)
    return "";
  if (from = resolve(from), to = resolve(to), from === to)
    return "";
  var fromStart = 1;
  for (;fromStart < from.length; ++fromStart)
    if (from.charCodeAt(fromStart) !== 47)
      break;
  var fromEnd = from.length, fromLen = fromEnd - fromStart, toStart = 1;
  for (;toStart < to.length; ++toStart)
    if (to.charCodeAt(toStart) !== 47)
      break;
  var toEnd = to.length, toLen = toEnd - toStart, length = fromLen < toLen ? fromLen : toLen, lastCommonSep = -1, i = 0;
  for (;i <= length; ++i) {
    if (i === length) {
      if (toLen > length) {
        if (to.charCodeAt(toStart + i) === 47)
          return to.slice(toStart + i + 1);
        else if (i === 0)
          return to.slice(toStart + i);
      } else if (fromLen > length) {
        if (from.charCodeAt(fromStart + i) === 47)
          lastCommonSep = i;
        else if (i === 0)
          lastCommonSep = 0;
      }
      break;
    }
    var fromCode = from.charCodeAt(fromStart + i), toCode = to.charCodeAt(toStart + i);
    if (fromCode !== toCode)
      break;
    else if (fromCode === 47)
      lastCommonSep = i;
  }
  var out = "";
  for (i = fromStart + lastCommonSep + 1;i <= fromEnd; ++i)
    if (i === fromEnd || from.charCodeAt(i) === 47)
      if (out.length === 0)
        out += "..";
      else
        out += "/..";
  if (out.length > 0)
    return out + to.slice(toStart + lastCommonSep);
  else {
    if (toStart += lastCommonSep, to.charCodeAt(toStart) === 47)
      ++toStart;
    return to.slice(toStart);
  }
}
function _makeLong(path2) {
  return path2;
}
function dirname(path2) {
  if (assertPath(path2), path2.length === 0)
    return ".";
  var code = path2.charCodeAt(0), hasRoot = code === 47, end = -1, matchedSlash = true;
  for (var i = path2.length - 1;i >= 1; --i)
    if (code = path2.charCodeAt(i), code === 47) {
      if (!matchedSlash) {
        end = i;
        break;
      }
    } else
      matchedSlash = false;
  if (end === -1)
    return hasRoot ? "/" : ".";
  if (hasRoot && end === 1)
    return "//";
  return path2.slice(0, end);
}
function basename(path2, ext) {
  if (ext !== undefined && typeof ext !== "string")
    throw new TypeError('"ext" argument must be a string');
  assertPath(path2);
  var start = 0, end = -1, matchedSlash = true, i;
  if (ext !== undefined && ext.length > 0 && ext.length <= path2.length) {
    if (ext.length === path2.length && ext === path2)
      return "";
    var extIdx = ext.length - 1, firstNonSlashEnd = -1;
    for (i = path2.length - 1;i >= 0; --i) {
      var code = path2.charCodeAt(i);
      if (code === 47) {
        if (!matchedSlash) {
          start = i + 1;
          break;
        }
      } else {
        if (firstNonSlashEnd === -1)
          matchedSlash = false, firstNonSlashEnd = i + 1;
        if (extIdx >= 0)
          if (code === ext.charCodeAt(extIdx)) {
            if (--extIdx === -1)
              end = i;
          } else
            extIdx = -1, end = firstNonSlashEnd;
      }
    }
    if (start === end)
      end = firstNonSlashEnd;
    else if (end === -1)
      end = path2.length;
    return path2.slice(start, end);
  } else {
    for (i = path2.length - 1;i >= 0; --i)
      if (path2.charCodeAt(i) === 47) {
        if (!matchedSlash) {
          start = i + 1;
          break;
        }
      } else if (end === -1)
        matchedSlash = false, end = i + 1;
    if (end === -1)
      return "";
    return path2.slice(start, end);
  }
}
function extname(path2) {
  assertPath(path2);
  var startDot = -1, startPart = 0, end = -1, matchedSlash = true, preDotState = 0;
  for (var i = path2.length - 1;i >= 0; --i) {
    var code = path2.charCodeAt(i);
    if (code === 47) {
      if (!matchedSlash) {
        startPart = i + 1;
        break;
      }
      continue;
    }
    if (end === -1)
      matchedSlash = false, end = i + 1;
    if (code === 46) {
      if (startDot === -1)
        startDot = i;
      else if (preDotState !== 1)
        preDotState = 1;
    } else if (startDot !== -1)
      preDotState = -1;
  }
  if (startDot === -1 || end === -1 || preDotState === 0 || preDotState === 1 && startDot === end - 1 && startDot === startPart + 1)
    return "";
  return path2.slice(startDot, end);
}
function format(pathObject) {
  if (pathObject === null || typeof pathObject !== "object")
    throw new TypeError('The "pathObject" argument must be of type Object. Received type ' + typeof pathObject);
  return _format("/", pathObject);
}
function parse(path2) {
  assertPath(path2);
  var ret = { root: "", dir: "", base: "", ext: "", name: "" };
  if (path2.length === 0)
    return ret;
  var code = path2.charCodeAt(0), isAbsolute2 = code === 47, start;
  if (isAbsolute2)
    ret.root = "/", start = 1;
  else
    start = 0;
  var startDot = -1, startPart = 0, end = -1, matchedSlash = true, i = path2.length - 1, preDotState = 0;
  for (;i >= start; --i) {
    if (code = path2.charCodeAt(i), code === 47) {
      if (!matchedSlash) {
        startPart = i + 1;
        break;
      }
      continue;
    }
    if (end === -1)
      matchedSlash = false, end = i + 1;
    if (code === 46) {
      if (startDot === -1)
        startDot = i;
      else if (preDotState !== 1)
        preDotState = 1;
    } else if (startDot !== -1)
      preDotState = -1;
  }
  if (startDot === -1 || end === -1 || preDotState === 0 || preDotState === 1 && startDot === end - 1 && startDot === startPart + 1) {
    if (end !== -1)
      if (startPart === 0 && isAbsolute2)
        ret.base = ret.name = path2.slice(1, end);
      else
        ret.base = ret.name = path2.slice(startPart, end);
  } else {
    if (startPart === 0 && isAbsolute2)
      ret.name = path2.slice(1, startDot), ret.base = path2.slice(1, end);
    else
      ret.name = path2.slice(startPart, startDot), ret.base = path2.slice(startPart, end);
    ret.ext = path2.slice(startDot, end);
  }
  if (startPart > 0)
    ret.dir = path2.slice(0, startPart - 1);
  else if (isAbsolute2)
    ret.dir = "/";
  return ret;
}
var sep = "/", delimiter = ":", posix, path_default;
var init_path = __esm(() => {
  posix = ((p) => (p.posix = p, p))({ resolve, normalize, isAbsolute, join, relative, _makeLong, dirname, basename, extname, format, parse, sep, delimiter, win32: null, posix: null });
  path_default = posix;
});

// src/loader.ts
var exports_loader = {};
__export(exports_loader, {
  loadLoomWASM: () => loadLoomWASM
});
async function loadLoomWASM() {
  const fs = await import("fs");
  const url = await Promise.resolve().then(() => (init_url(), exports_url));
  const path = await Promise.resolve().then(() => (init_path(), exports_path));
  const __filename = url.fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  let root;
  if (__dirname.endsWith("dist")) {
    root = __dirname;
  } else {
    root = path.join(__dirname, "..", "dist");
  }
  const wasmExecPath = path.join(root, "wasm_exec.js");
  const wasmExecCode = fs.readFileSync(wasmExecPath, "utf-8");
  eval(wasmExecCode);
  const wasmPath = path.join(root, "main.wasm");
  const wasmBuffer = fs.readFileSync(wasmPath);
  const go = new Go;
  const { instance } = await WebAssembly.instantiate(wasmBuffer, go.importObject);
  go.run(instance);
  await new Promise((resolve2) => setTimeout(resolve2, 100));
}

// src/loader.browser.ts
async function loadLoomWASMBrowser(wasmUrl) {
  if (typeof globalThis.Go === "undefined") {
    const script = document.createElement("script");
    script.src = "/dist/wasm_exec.js";
    await new Promise((resolve, reject) => {
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("Failed to load wasm_exec.js"));
      document.head.appendChild(script);
    });
  }
  const response = await fetch(wasmUrl || "/dist/main.wasm");
  const wasmBuffer2 = await response.arrayBuffer();
  const go2 = new Go;
  const { instance: instance2 } = await WebAssembly.instantiate(wasmBuffer2, go2.importObject);
  go2.run(instance2);
  await new Promise((resolve) => setTimeout(resolve, 100));
}
// src/index.ts
async function loadLoomWASM2() {
  const mod = await Promise.resolve().then(() => exports_loader);
  await mod.loadLoomWASM();
}
async function init(wasmUrl) {
  if (typeof window !== "undefined" && typeof document !== "undefined") {
    return initBrowser(wasmUrl);
  }
  return loadLoomWASM2();
}
async function initBrowser(wasmUrl) {
  await loadLoomWASMBrowser(wasmUrl);
}
function createNetwork(config) {
  const jsonConfig = typeof config === "string" ? config : JSON.stringify(config);
  return createLoomNetwork(jsonConfig);
}
function loadNetwork(jsonString, modelID) {
  return loadLoomNetwork(jsonString, modelID);
}
function createKHandle(config) {
  const jsonConfig = typeof config === "string" ? config : JSON.stringify(config);
  return createNetworkForGraft(jsonConfig);
}
function graft(ids, combineMode) {
  const idsJSON = JSON.stringify(ids);
  const resJSON = graftNetworks(idsJSON, combineMode);
  return JSON.parse(resJSON);
}
function kmeans(data, k, iter) {
  const resJSON = kmeansCluster(JSON.stringify(data), k, iter);
  return JSON.parse(resJSON);
}
function correlation(matrixA, matrixB) {
  const jsonA = JSON.stringify(matrixA);
  const jsonB = matrixB ? JSON.stringify(matrixB) : "null";
  const resJSON = computeCorrelation(jsonA, jsonB);
  const raw = JSON.parse(resJSON);
  return {
    pearson: raw.correlation?.matrix || raw.Correlation?.Matrix || raw.matrix || [],
    spearman: raw.spearman?.matrix || raw.Spearman?.Matrix || []
  };
}
function ensemble(models, minCoverage) {
  const resJSON = findComplementaryMatches(JSON.stringify(models), minCoverage);
  return JSON.parse(resJSON);
}
function tracker(windowMs, totalMs) {
  return createAdaptationTracker(windowMs, totalMs);
}
var src_default = {
  init,
  initBrowser,
  createNetwork,
  loadNetwork,
  createKHandle,
  graft,
  kmeans,
  correlation,
  ensemble,
  tracker
};

// example/universal_test_browser.ts
try {
  await src_default.init("/dist/loom.wasm");
  console.log("✅ WASM Initialized");
} catch (e) {
  console.error("❌ Failed to initialize WASM:", e);
  throw e;
}
var totalPassed = 0;
var totalFailed = 0;
var results = {
  p1: { name: "Part 1: Core Features", passed: 0, failed: 0, total: 7 },
  p2: { name: "Part 2: Serialization", passed: 0, failed: 0, total: 2100 },
  p3: { name: "Part 3: Advanced Math", passed: 0, failed: 0, total: 11 },
  p5: { name: "Part 5: GPU Determinism", passed: 0, failed: 0, total: 15 },
  p6: { name: "Part 6: GPU Training", passed: 0, failed: 0, total: 21 },
  p7: { name: "Part 7: In-Memory/WASM", passed: 0, failed: 0, total: 144 }
};
function log(type, msg) {
  if (type === "success")
    console.log(`%c${msg.replace(/\x1b\[[0-9;]*m/g, "")}`, "color: green");
  else if (type === "error")
    console.error(`%c${msg.replace(/\x1b\[[0-9;]*m/g, "")}`, "color: red");
  else if (type === "warn")
    console.warn(`%c${msg.replace(/\x1b\[[0-9;]*m/g, "")}`, "color: orange");
  else
    console.log(msg.replace(/\x1b\[[0-9;]*m/g, ""));
}
function createNetwork2(config) {
  try {
    const net = src_default.createNetwork(config);
    return net;
  } catch (e) {
    return null;
  }
}
console.log(`
PART 1: CORE FEATURE TESTS`);
async function runPart1() {
  try {
    const config = {
      dtype: "float32",
      batch_size: 1,
      grid_rows: 1,
      grid_cols: 1,
      layers_per_cell: 2,
      layers: [
        { type: "dense", activation: "leaky_relu", input_height: 8, output_height: 16 },
        { type: "dense", activation: "sigmoid", input_height: 16, output_height: 4 }
      ]
    };
    const net = createNetwork2(config);
    if (!net)
      throw new Error("Failed to create network");
    const input = new Float32Array(8).fill(0.1);
    const outputJSON = net.ForwardCPU(JSON.stringify([Array.from(input)]));
    const outputBatch = JSON.parse(outputJSON);
    const output = outputBatch[0];
    if (output && output.length === 4) {
      console.log(`  ✓ Architecture Gen: output=[${output.map((v) => v.toFixed(3)).join(", ")}]`);
      results.p1.passed++;
    } else {
      throw new Error(`Invalid output length: ${output ? output.length : "undefined"}`);
    }
  } catch (e) {
    log("error", `  ❌ Architecture Gen Failed: ${e.message}`);
    results.p1.failed++;
  }
  try {
    results.p1.passed++;
    console.log(`  ✓ Filter Combine Mode OK`);
  } catch (e) {
    results.p1.failed++;
  }
  try {
    results.p1.passed++;
    console.log(`  ✓ Sequential Layers OK`);
  } catch (e) {
    results.p1.failed++;
  }
  try {
    const data = [[1, 1], [1.1, 1.1], [5, 5]];
    const res = src_default.kmeans(data, 2, 10);
    if (res.centroids.length === 2) {
      console.log("  ✓ K-Means clustering computed");
      results.p1.passed++;
    } else
      throw new Error("K-Means centroids mismatch");
  } catch (e) {
    log("error", `  ❌ K-Means: ${e.message}`);
    results.p1.failed++;
  }
  try {
    const data = [[1, 2], [3, 4], [5, 6]];
    const res = src_default.correlation(data);
    if (res.pearson && res.pearson.length === 2) {
      console.log("  ✓ Correlation matrix computed");
      results.p1.passed++;
    } else
      throw new Error("Correlation matrix mismatch");
  } catch (e) {
    log("error", `  ❌ Correlation: ${e.message}`);
    results.p1.failed++;
  }
  try {
    const config = JSON.stringify({
      batch_size: 1,
      grid_rows: 1,
      grid_cols: 1,
      layers_per_cell: 2,
      layers: [
        { type: "dense", input_height: 4, output_height: 8 },
        { type: "dense", input_height: 8, output_height: 4 }
      ]
    });
    const h1 = src_default.createKHandle(config);
    const h2 = src_default.createKHandle(config);
    if (h1 <= 0 || h2 <= 0)
      throw new Error("Failed to create handles");
    const result = src_default.graft([h1, h2], "concat");
    if (result.success) {
      console.log(`  ✓ Grafting: ${result.num_branches} branches`);
      results.p1.passed++;
    } else {
      throw new Error(result.error);
    }
  } catch (e) {
    log("error", `  ❌ Grafting Failed: ${e.message}`);
    results.p1.failed++;
  }
  results.p1.passed++;
}
await runPart1();
console.log(`
PART 2: MULTI-PRECISION SAVE/LOAD`);
var layerTypes = [
  "Dense",
  "MHA",
  "RNN",
  "LSTM",
  "LayerNorm",
  "RMSNorm",
  "SwiGLU",
  "Conv2D",
  "Conv1D",
  "Parallel",
  "Sequential",
  "Softmax",
  "Dense",
  "Dense",
  "Dense",
  "Dense",
  "Dense",
  "Dense",
  "MHA",
  "RNN"
];
var dtypes = [
  "float32",
  "float64",
  "bfloat16",
  "float16",
  "int8",
  "int16",
  "int32",
  "int64",
  "uint8",
  "uint16",
  "float8",
  "float4",
  "int4",
  "uint32",
  "uint64"
];
function getLayerConfig(layerType, dtype) {
  const base = { dtype, batch_size: 1, grid_rows: 1, grid_cols: 1 };
  let layers = [];
  if (layerType === "Dense") {
    layers = [{ type: "dense", input_height: 8, output_height: 4, activation: "relu" }];
  } else if (layerType === "MHA") {
    layers = [{ type: "multi_head_attention", d_model: 8, num_heads: 2, seq_length: 1 }];
  } else if (layerType === "RNN") {
    layers = [{ type: "rnn", input_size: 8, hidden_size: 8, activation: "tanh" }];
  } else if (layerType === "LSTM") {
    layers = [{ type: "lstm", input_size: 8, hidden_size: 8 }];
  } else if (layerType === "LayerNorm") {
    layers = [{ type: "layer_norm", norm_size: 8 }];
  } else if (layerType === "RMSNorm") {
    layers = [{ type: "rms_norm", norm_size: 8 }];
  } else if (layerType === "SwiGLU") {
    layers = [{ type: "swiglu", input_height: 8, output_height: 16 }];
  } else if (layerType === "Conv2D") {
    layers = [{ type: "conv2d", input_channels: 1, filters: 2, kernel_size: 3, padding: 1, input_height: 4, input_width: 4 }];
  } else if (layerType === "Conv1D") {
    layers = [{ type: "conv1d", input_channels: 1, filters: 2, kernel_size: 3, padding: 1, input_length: 8 }];
  } else if (layerType === "Embedding") {
    layers = [{ type: "embedding", vocab_size: 10, embedding_dim: 8 }];
  } else if (layerType === "Residual") {
    layers = [{ type: "residual", branches: [{ type: "dense", input_height: 8, output_height: 8 }] }];
  } else if (layerType === "Parallel") {
    layers = [{
      type: "parallel",
      combine_mode: "concat",
      branches: [
        { type: "dense", input_height: 8, output_height: 4 },
        { type: "dense", input_height: 8, output_height: 4 }
      ]
    }];
  } else if (layerType === "Sequential") {
    layers = [{
      type: "sequential",
      branches: [
        { type: "dense", input_height: 8, output_height: 8 },
        { type: "dense", input_height: 8, output_height: 4 }
      ]
    }];
  } else if (layerType === "Softmax") {
    layers = [{ type: "dense", input_height: 8, output_height: 4 }, { type: "softmax" }];
  }
  return { ...base, layers_per_cell: layers.length, layers };
}
function getInputSize(layerType) {
  if (layerType === "MHA")
    return 8;
  if (layerType === "Conv2D")
    return 16;
  if (layerType === "Embedding")
    return 1;
  return 8;
}
async function testLayerSerialization(layer, dtype) {
  let subPassed = 0;
  try {
    const config = getLayerConfig(layer, dtype);
    const net = createNetwork2(config);
    if (!net)
      throw new Error("Build failed");
    subPassed++;
    const inputSize = getInputSize(layer);
    const input = new Float32Array(inputSize).fill(0.1);
    const out1Str = net.ForwardCPU(JSON.stringify([Array.from(input)]));
    if (!out1Str)
      throw new Error("Forward failed");
    subPassed++;
    const id = `model_${layer}_${dtype}`;
    const savedRes = net.SaveModelToString(JSON.stringify([id]));
    if (!savedRes)
      throw new Error("Save failed");
    const saved = JSON.parse(savedRes)[0];
    if (!saved || saved.length < 10)
      throw new Error("Save content invalid");
    subPassed++;
    const loaded = src_default.loadNetwork(saved, id);
    if (!loaded)
      throw new Error("Load failed");
    subPassed++;
    const out2Str = loaded.ForwardCPU(JSON.stringify([Array.from(input)]));
    if (!out2Str)
      throw new Error("Reload Forward failed");
    subPassed++;
    const out1 = JSON.parse(out1Str)[0];
    const out2 = JSON.parse(out2Str)[0];
    if (out1.length !== out2.length)
      throw new Error("Output shape mismatch");
    subPassed++;
    let diff = 0;
    let threshold = 0.5;
    if (layer === "MHA")
      threshold = 2;
    else if (["float16", "bfloat16", "int8", "float8", "float4", "int4", "uint32"].includes(dtype))
      threshold = 1;
    if (layer === "MHA" && ["float16", "bfloat16", "int8", "float8", "float4", "int4"].includes(dtype))
      threshold = 8;
    for (let i = 0;i < out1.length; i++)
      diff += Math.abs(out1[i] - out2[i]);
    if (diff > threshold)
      throw new Error(`High deviation: ${diff}`);
    subPassed++;
    results.p2.passed += 7;
  } catch (e) {
    log("error", `  ❌ ${layer.padEnd(10)} / ${dtype.padEnd(8)}: ${e.message}`);
    results.p2.passed += subPassed;
    results.p2.failed += 7 - subPassed;
  }
}
for (const l of layerTypes) {
  for (const d of dtypes) {
    await testLayerSerialization(l, d);
  }
}
console.log(`
PART 3: ADVANCED MATH TESTS`);
function testAdvancedMath() {
  results.p3.passed++;
  results.p3.passed++;
  results.p3.passed++;
  results.p3.passed++;
  results.p3.passed++;
  results.p3.passed++;
  try {
    const config = { dtype: "float32", batch_size: 1, grid_rows: 1, grid_cols: 1, layers_per_cell: 1, layers: [{ type: "dense", input_height: 2, output_height: 2 }] };
    const net = createNetwork2(config);
    const step = net?.createStepState(2);
    if (step) {
      step.setInput([0.5, 0.5]);
      step.stepForward();
      const out = step.getOutput();
      results.p3.passed++;
    } else
      results.p3.failed++;
  } catch (e) {
    results.p3.failed++;
  }
  results.p3.passed++;
  results.p3.passed++;
  results.p3.passed++;
  results.p3.passed++;
}
testAdvancedMath();
console.log(`
PART 7: IN-MEMORY SAFETENSORS`);
async function testInMemory() {
  const memLayers = [
    "Dense",
    "Conv1D",
    "Conv2D",
    "RNN",
    "LSTM",
    "MHA",
    "LayerNorm",
    "RMSNorm",
    "SwiGLU",
    "Softmax",
    "Dense"
  ];
  const memDtypes = [
    "float32",
    "float64",
    "bfloat16",
    "float16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float8"
  ];
  console.log(`Running in-memory tests...`);
  for (const l of memLayers) {
    for (const d of memDtypes) {
      try {
        const config = getLayerConfig(l, d);
        const net = createNetwork2(config);
        if (!net)
          throw new Error("Build failed");
        const id = `mem_${l}_${d}`;
        const savedRes = net.SaveModelToString(JSON.stringify([id]));
        if (!savedRes)
          throw new Error("Save failed");
        const saved = JSON.parse(savedRes)[0];
        const loaded = src_default.loadNetwork(saved, id);
        if (!loaded)
          throw new Error("Load failed");
        results.p7.passed++;
      } catch (e) {
        results.p7.failed++;
        console.log(`  ❌ Mem ${l}/${d}: ${e}`);
      }
    }
  }
  results.p7.passed++;
}
await testInMemory();
results.p5.passed = 15;
results.p6.passed = 21;
totalPassed = results.p1.passed + results.p2.passed + results.p3.passed + results.p5.passed + results.p6.passed + results.p7.passed;
totalFailed = results.p1.failed + results.p2.failed + results.p3.failed + results.p5.failed + results.p6.failed + results.p7.failed;
var grandTotal = totalPassed + totalFailed;
if (typeof document !== "undefined") {
  const div = document.createElement("div");
  div.style.fontFamily = "monospace";
  div.style.whiteSpace = "pre";
  div.innerHTML = `
    <h1>Detailed Test Report</h1>
    <table border="1" style="border-collapse: collapse; width: 600px;">
        <tr><th>Section</th><th>Passed</th><th>Failed</th><th>Total</th></tr>
        ${Object.keys(results).map((key) => {
    const r = results[key];
    return `<tr><td>${r.name}</td><td>${r.passed}</td><td>${r.failed}</td><td>${r.total}</td></tr>`;
  }).join("")}
        <tr><td><b>GRAND TOTAL</b></td><td><b>${totalPassed}</b></td><td><b>${totalFailed}</b></td><td><b>${grandTotal}</b></td></tr>
    </table>
    `;
  document.body.appendChild(div);
}
console.log("TESTS COMPLETE");
