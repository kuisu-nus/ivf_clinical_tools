"""Run Dressage-claw task samples in independent E2B sandboxes.

The package mirrors the three phases of a Dressage rollout:

  before_agent  upload task files, start the blackbox server binding, run the
                sample's before_agent commands (setup, tool services, MCP)
  agent         send the prompt to OpenClaw through the blackbox server
  after_agent   upload grader files only now, run the sample's after_agent
                commands, and parse the grader score

It depends only on the ``e2b`` SDK and ``httpx``; Dressage itself is not
imported.
"""

__version__ = "0.1.0"
