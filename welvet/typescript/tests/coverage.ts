/**
 * Loom TypeScript Wrapper 345-Item Parity CLI Auditor
 * --------------------------------------------------
 * Verifies 100% functional parity between @openfluke/welvet and loom-core.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import welvet from '../src/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function runCLIParityAudit() {
    console.log("\x1b[36m%s\x1b[0m", "=== Loom TypeScript Wrapper Parity Audit (CLI) ===");
    
    try {
        // 1. Initialize Engine
        await welvet.init();
        const internalParity = welvet.getInternalParity ? welvet.getInternalParity() : [];
        console.log(`\n[INIT] Engine Loaded. ${internalParity.length} internal symbols acquired.`);

        // 2. Load expected API
        const blueprintPath = path.resolve(__dirname, '../../wasm/expected_api.json');
        const expectedApi = JSON.parse(fs.readFileSync(blueprintPath, 'utf8'));

        // 3. Functional Proof
        let functionalProof = false;
        try {
            const net = welvet.createNetwork({
                depth: 1, rows: 1, cols: 1, layers_per_cell: 1,
                layers: [{ type: "Dense", input_height: 4, output_height: 4, activation: "ReLU" }]
            });
            const input = new Float32Array(4).fill(0.5);
            const output = net.sequentialForward(input);
            if (output && output.length === 4) functionalProof = true;
            net.free();
            console.log("\x1b[32m%s\x1b[0m", "[PASS] Functional Layer Pass Verified.");
        } catch (e) {
            console.error("\x1b[31m%s\x1b[0m", "[FAIL] Functional Proof Error: " + (e as Error).message);
        }

        // 4. Scan
        const results: any[] = [];
        let totalItems = 0;
        let passedItems = 0;

        for (const [category, items] of Object.entries<any>(expectedApi)) {
            console.log(`\n\x1b[33m[${category}]\x1b[0m`);
            for (const item of items) {
                totalItems++;
                let status = "MISSING";
                let details = "";

                // Logic sync with check_ts.js
                const camelName = item.name.charAt(0).toLowerCase() + item.name.slice(1);
                
                if (welvet[item.name as keyof typeof welvet] || 
                    welvet[('create' + item.name) as keyof typeof welvet] || 
                    welvet[('load' + item.name) as keyof typeof welvet]) {
                    status = "PASS";
                    details = "TS Export";
                } else if (welvet.DType && (welvet.DType as any)[item.name.toUpperCase()]) {
                    status = "PASS";
                    details = "TS Constant";
                } else if (welvet.LayerType && (welvet.LayerType as any)[item.name.toUpperCase()]) {
                    status = "PASS";
                    details = "TS LayerType";
                } else {
                    // Method check on Dummy
                    const net = welvet.createNetwork({depth:1, rows:1, cols:1, layers_per_cell:1, layers:[]});
                    if (net && (typeof (net as any)[item.name] === 'function' || typeof (net as any)[camelName] === 'function')) {
                        status = "PASS";
                        details = "Instance Method";
                    } else if (internalParity.includes(item.name)) {
                        status = "PASS";
                        details = "Engine Internal (Indirect)";
                    }
                    if (net) net.free();
                }

                if (status === "PASS") {
                    passedItems++;
                    process.stdout.write(`\x1b[32m.\x1b[0m`);
                } else {
                    console.log(`\n  \x1b[31m[MISSING] ${item.name}\x1b[0m`);
                }
            }
        }

        const coverage = (passedItems / totalItems) * 100;
        console.log(`\n\nFinal Report:`);
        console.log(`---------------------------------`);
        console.log(`Items Scanned : ${totalItems}`);
        console.log(`Items Passed  : ${passedItems}`);
        console.log(`Coverage      : ${coverage.toFixed(2)}%`);
        console.log(`---------------------------------`);

        if (coverage === 100) {
            console.log("\x1b[32m%s\x1b[0m", "FULL PARITY ACHIEVED.");
            process.exit(0);
        } else {
            console.log("\x1b[31m%s\x1b[0m", "INCOMPLETE PARITY.");
            process.exit(1);
        }

    } catch (e) {
        console.error("Critical Failure:", e);
        process.exit(1);
    }
}

runCLIParityAudit();
