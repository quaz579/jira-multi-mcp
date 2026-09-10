# jira-multi-mcp

A single MCP server that talks to several Jira Cloud sites at once. It spawns
the upstream [`sooperset/mcp-atlassian`](https://github.com/sooperset/mcp-atlassian)
server unmodified, one child process per configured site, mirrors its tools
behind a `site` selector so a caller can just say "look at JUMP-2274" and get
routed by issue-key prefix, and adds attachment download/upload tools that
write straight to disk instead of returning base64.

Status: pre-alpha, under construction. This repo currently ships config
loading, the site registry, and a CLI (`--check`, `--print-config`, `--warm`);
`serve` (the actual MCP server) lands in a later milestone.

## Configuration

Copy `config.example.toml` to `${XDG_CONFIG_HOME:-~/.config}/jira-multi-mcp/config.toml`
(`chmod 600` recommended) and fill in your sites. Override the location with
`JIRA_MULTI_CONFIG=/path/to/config.toml` or `--config PATH`. Each `[[sites]]`
entry needs a unique `name`, an `https://` `url`, and a globally-unique set of
`key_prefixes` (the project prefixes, e.g. `JMC`, used to route a bare
`JMC-1234` to the right site without saying which one). Auth is either Cloud
(`username` + `api_token`/`api_token_env`) or Server/Data Center
(`personal_token`/`personal_token_env`) — see `config.example.toml` for the
full contract, including the `[defaults]` fallbacks and the
`JIRA_MULTI_SITE_<NAME>_<FIELD>` environment-variable overlay.

Useful commands:

- `jira-multi-mcp --print-config` — show the effective, merged configuration
  with every secret masked as `***`.
- `jira-multi-mcp --check` — authenticate against every configured site and
  print a table of `displayName`/`accountId` per site; exits non-zero if any
  site fails (`--allow-partial` to tolerate some failures).
- `jira-multi-mcp --warm [--refresh]` — prime the `uvx` cache for the
  upstream `mcp-atlassian` command ahead of time.
