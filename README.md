# jira-multi-mcp

A single MCP server that talks to several Jira Cloud sites at once. It spawns
the upstream [`sooperset/mcp-atlassian`](https://github.com/sooperset/mcp-atlassian)
server unmodified, one child process per configured site, mirrors its tools
behind a `site` selector so a caller can just say "look at JUMP-2274" and get
routed by issue-key prefix, and adds attachment download/upload tools that
write straight to disk instead of returning base64.

Status: pre-alpha, under construction. No usable server yet; this repo
currently ships only a CLI skeleton and CI.
