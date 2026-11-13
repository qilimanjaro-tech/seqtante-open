# Issue tracker: Linear, team QHC

Work on seqtante-open is tracked in Linear, team QHC. Tickets are `QHC-####`. GitHub Issues is not the work queue, although the repository lives on GitHub.

## Conventions

- **Read or search a ticket:** use the Linear tools (for example the Linear MCP server's `list_issues` and `get_issue`) scoped to team QHC, and refer to tickets by their identifier (`QHC-1798`).
- **Create or update a ticket:** use the Linear tools on team QHC. Show the human the title and body and get a yes before creating or editing one.
- **Branches and PRs:** branch names derive from the ticket (`qhc-1798-<topic>`), and PR titles read `[QHC-####] Name` with the ticket linked. See `## Branches and PRs` in `AGENTS.md`.
- **Do not** use `gh issue create` for seqtante-open work.

## GitHub issues from outside users

The repository is public, so people outside the team may open GitHub issues. QHC's weekly intake reads them into the team's request queue, and triage decides which become Linear tickets. Work starts from a Linear ticket, not from the GitHub issue.

- Read a GitHub issue with `gh issue view <number> --comments` when a ticket links one.
- Don't comment on, label or close a GitHub issue without the human's yes. Replies to outside users are the team's to write.

## Pull requests as a triage surface

Off. External pull requests are not part of the triage queue. Flip this if the team starts triaging drive-by PRs.

## When a skill says "publish to the issue tracker"

Create a Linear ticket on team QHC, after the human approves its title and body.

## When a skill says "fetch the relevant ticket"

Read the Linear ticket by its `QHC-####` identifier, as in **Read or search a ticket** above.
