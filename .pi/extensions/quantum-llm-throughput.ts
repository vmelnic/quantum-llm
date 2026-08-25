import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

type ActiveResponse = {
	startedAt: number;
	firstDeltaAt?: number;
};

const PROVIDER = "quantum-llm";
const STATUS_KEY = "quantum-llm-throughput";

function seconds(milliseconds: number): string {
	const value = milliseconds / 1000;
	return value < 10 ? value.toFixed(2) : value.toFixed(1);
}

export default function (pi: ExtensionAPI) {
	let active: ActiveResponse | undefined;

	pi.on("message_start", async (event, ctx) => {
		if (event.message.role !== "assistant" ||
			event.message.provider !== PROVIDER) {
			return;
		}
		active = { startedAt: Date.now() };
		ctx.ui.setStatus(STATUS_KEY, "Quantum LLM: generating…");
	});

	pi.on("message_update", async (event) => {
		if (!active || event.message.role !== "assistant" ||
			event.message.provider !== PROVIDER) {
			return;
		}
		const update = event.assistantMessageEvent;
		if ((update.type === "text_delta" ||
				update.type === "thinking_delta" ||
				update.type === "toolcall_delta") &&
			update.delta.length > 0 && active.firstDeltaAt === undefined) {
			active.firstDeltaAt = Date.now();
		}
	});

	pi.on("message_end", async (event, ctx) => {
		if (!active || event.message.role !== "assistant" ||
			event.message.provider !== PROVIDER) {
			return;
		}
		const endedAt = Date.now();
		const outputTokens = event.message.usage.output;
		const totalMs = Math.max(0, endedAt - active.startedAt);
		const fields: string[] = [];
		if (active.firstDeltaAt !== undefined) {
			const decodeMs = endedAt - active.firstDeltaAt;
			if (outputTokens > 1 && decodeMs > 0) {
				const rate = (outputTokens - 1) / (decodeMs / 1000);
				fields.push(`${rate.toFixed(2)} tok/s decode`);
			}
			fields.push(
				`TTFT ${seconds(active.firstDeltaAt - active.startedAt)}s`,
			);
		}
		fields.push(`${outputTokens} tok`, `${seconds(totalMs)}s total`);
		ctx.ui.setStatus(STATUS_KEY, `Quantum LLM: ${fields.join(" · ")}`);
		active = undefined;
	});
}
