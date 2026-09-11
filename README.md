# jira-multi-mcp

A single MCP server that talks to several Jira Cloud sites at once. It spawns
the upstream [`sooperset/mcp-atlassian`](https://github.com/sooperset/mcp-atlassian)
server unmodified, one child process per configured site, mirrors its tools
behind a `site` selector so a caller can just say "look at JUMP-2274" and get
routed by issue-key prefix, and adds attachment download/upload tools that
write straight to disk instead of returning base64.

Status: pre-alpha, under construction. This repo currently ships config
loading, the site registry, a CLI (`--check`, `--print-config`, `--warm`), the
running `serve` MCP server (child processes mirrored behind a `site`
selector), and the attachment tools described below.

## Attachment tools

Attachments are implemented directly against Jira Cloud REST v3 (not
forwarded to an upstream child), because upstream `mcp-atlassian` only
returns attachment content as base64 in-band. Jira Cloud sites only in this
version — a `personal_token` (Server/Data Center) site gets a clear refusal
instead.

- `jira_list_attachments(issue_key, site?)` — `id`, `filename`, `size`,
  `mime_type`, `created`, `author` for every attachment on the issue.
- `jira_download_attachments(issue_key, target_dir, site?, filenames?, attachment_ids?, overwrite?)`
  — writes attachments to `target_dir` and returns their paths (read the file
  from disk afterward; prefer this over any base64 tool). Omit
  `filenames`/`attachment_ids` to download everything. An existing file is
  never silently overwritten: a same-named download falls back to
  `{name}-{attachment_id}{ext}`, and if that also exists it's reported under
  `skipped` rather than written (`overwrite=true` to replace in place
  instead). Anything selected but not written lands in `skipped` (name
  collision) or `failed` (rejected by Jira, over `defaults.attachment_max_bytes`,
  or an unsafe filename).
- `jira_upload_attachments(issue_key, paths, site?)` — uploads local files
  (each must already exist as a regular file) and returns the created
  attachments' `id`/`filename`/`size`/`mime_type`.

`site` is optional on all three and inferred from `issue_key`'s project
prefix the same way every mirrored tool resolves it.

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
