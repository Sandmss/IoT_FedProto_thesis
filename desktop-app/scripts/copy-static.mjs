import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const sourceDir = path.join(projectRoot, "src", "renderer");
const targetDir = path.join(projectRoot, "dist", "src", "renderer");

mkdirSync(targetDir, { recursive: true });
writeFileSync(path.join(targetDir, "index.html"), readFileSync(path.join(sourceDir, "index.html")));
writeFileSync(path.join(targetDir, "styles.css"), readFileSync(path.join(sourceDir, "styles.css")));
