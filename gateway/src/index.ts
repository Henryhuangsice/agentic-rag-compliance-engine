import { resolve } from "node:path";
import { loadConfig } from "./config";
import { buildServer } from "./server";

const config = loadConfig();
if (config.webDist) config.webDist = resolve(config.webDist);
const app = await buildServer(config, { logger: true });

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => void app.close().then(() => process.exit(0)));
}
await app.listen({ host: config.host, port: config.port });
