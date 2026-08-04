# Changelog

All notable project changes are recorded here. The project follows semantic
versioning once public releases begin.

## 0.1.0 — 2026-08-04

- added deterministic Expert Pack v1 compilation for OLMoE and Qwen3-Next;
- implemented bounded SSD/RAM/VRAM expert placement and CUDA SM86 execution;
- completed Qwen3-Next attention, DeltaNet, MoE, and multi-request state;
- added continuous decode batching and OpenAI-compatible text endpoints;
- qualified the Qwen3-Next 80B hot path on one RTX 3090;
- published the controlled 4096-context pre-production pilot profile;
- documented long-context, cold-latency, service-hardening, Metal, and
  distributed-expert work still required.

