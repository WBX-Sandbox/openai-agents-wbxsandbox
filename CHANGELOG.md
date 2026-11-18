# Changelog

## 0.1.0 – 2026-11-18

First release.

- `WBXSandboxClient`, `WBXSandboxClientOptions`, `WBXSandboxSession`, `WBXSandboxSessionState`
  for OpenAI Agents SDK sandbox agents (`openai-agents` 0.25.x).
- Commands, file reads and writes, workspace snapshots (tar) and restore, resume of saved run
  state, exposed ports as HTTPS URLs with an access token.
- `on_exit="sleep"` keeps the sandbox's files after the run so a later resume continues in the
  same workspace; named sandboxes (`name=`) are reused across runs.
- Network policy per sandbox: no network, allowlist, blocklist, IPv4/IPv6.
- Not supported yet: manifest mounts (S3, GCS, …), PTY sessions.
