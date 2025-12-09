[Skip to main content](#content-area)
🚀 [Share how you're building agents](https://bit.ly/agent-engineering-survey) for a chance to win LangChain swag!
[Docs by LangChain home page![light logo](https://mintcdn.com/langchain-5e9cc07a/Xbr8HuVd9jPi6qTU/images/brand/langchain-docs-teal.svg?fit=max&auto=format&n=Xbr8HuVd9jPi6qTU&q=85&s=16111530672bf976cb54ef2143478342)![dark logo](https://mintcdn.com/langchain-5e9cc07a/Xbr8HuVd9jPi6qTU/images/brand/langchain-docs-lilac.svg?fit=max&auto=format&n=Xbr8HuVd9jPi6qTU&q=85&s=b70fb1a2208670492ef94aef14b680be)](/)LangChain + LangGraph
Search...
⌘K
* [Ask AI](https://chat.langchain.com/)
* [GitHub](https://github.com/langchain-ai)
* [Try LangSmith](https://smith.langchain.com/)
* [Try LangSmith](https://smith.langchain.com/)
Search...
Navigation
LangChain overview
[LangChain](/oss/python/langchain/overview)[LangGraph](/oss/python/langgraph/overview)[Deep Agents](/oss/python/deepagents/overview)[Integrations](/oss/python/integrations/providers/overview)[Learn](/oss/python/learn)[Reference](/oss/python/reference/overview)[Contribute](/oss/python/contributing/overview)
Python
* [Overview](/oss/python/langchain/overview)
##### Get started
* [Install](/oss/python/langchain/install)
* [Quickstart](/oss/python/langchain/quickstart)
* [Changelog](https://docs.langchain.com/oss/python/releases/changelog)
* [Philosophy](/oss/python/langchain/philosophy)
##### Core components
* [Agents](/oss/python/langchain/agents)
* [Models](/oss/python/langchain/models)
* [Messages](/oss/python/langchain/messages)
* [Tools](/oss/python/langchain/tools)
* [Short-term memory](/oss/python/langchain/short-term-memory)
* [Streaming](/oss/python/langchain/streaming)
* [Structured output](/oss/python/langchain/structured-output)
##### Middleware
* [Overview](/oss/python/langchain/middleware/overview)
* [Built-in middleware](/oss/python/langchain/middleware/built-in)
* [Custom middleware](/oss/python/langchain/middleware/custom)
##### Advanced usage
* [Guardrails](/oss/python/langchain/guardrails)
* [Runtime](/oss/python/langchain/runtime)
* [Context engineering](/oss/python/langchain/context-engineering)
* [Model Context Protocol (MCP)](/oss/python/langchain/mcp)
* [Human-in-the-loop](/oss/python/langchain/human-in-the-loop)
* [Multi-agent](/oss/python/langchain/multi-agent)
* [Retrieval](/oss/python/langchain/retrieval)
* [Long-term memory](/oss/python/langchain/long-term-memory)
##### Agent development
* [LangSmith Studio](/oss/python/langchain/studio)
* [Test](/oss/python/langchain/test)
* [Agent Chat UI](/oss/python/langchain/ui)
##### Deploy with LangSmith
* [Deployment](/oss/python/langchain/deploy)
* [Observability](/oss/python/langchain/observability)
On this page
* [Create an agent](#create-an-agent)
* [Core benefits](#core-benefits)
# LangChain overview
Copy page
Copy page
LangChain is the easiest way to start building agents and applications powered by LLMs. With under 10 lines of code, you can connect to OpenAI, Anthropic, Google, and [more](/oss/python/integrations/providers/overview). LangChain provides a pre-built agent architecture and model integrations to help you get started quickly and seamlessly incorporate LLMs into your agents and applications.
We recommend you use LangChain if you want to quickly build agents and autonomous applications. Use [LangGraph](/oss/python/langgraph/overview), our low-level agent orchestration framework and runtime, when you have more advanced needs that require a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency.
LangChain [agents](/oss/python/langchain/agents) are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. You do not need to know LangGraph for basic LangChain agent usage.
## [​](#create-an-agent) Create an agent
Copy
```
# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```
See the [Installation instructions](/oss/python/langchain/install) and [Quickstart guide](/oss/python/langchain/quickstart) to get started building your own agents and applications with LangChain.
## [​](#core-benefits) Core benefits
[## Standard model interface
Different providers have unique APIs for interacting with models, including the format of responses. LangChain standardizes how you interact with models so that you can seamlessly swap providers and avoid lock-in.
Learn more](/oss/python/langchain/models)[## Easy to use, highly flexible agent
LangChain’s agent abstraction is designed to be easy to get started with, letting you build a simple agent in under 10 lines of code. But it also provides enough flexibility to allow you to do all the context engineering your heart desires.
Learn more](/oss/python/langchain/agents)[## Built on top of LangGraph
LangChain’s agents are built on top of LangGraph. This allows us to take advantage of LangGraph’s durable execution, human-in-the-loop support, persistence, and more.
Learn more](/oss/python/langgraph/overview)[## Debug with LangSmith
Gain deep visibility into complex agent behavior with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.
Learn more](/langsmith/home)
---
[Edit the source of this page on GitHub.](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/overview.mdx)
[Connect these docs programmatically](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.
Was this page helpful?
YesNo
[Install LangChain
Next](/oss/python/langchain/install)
⌘I
[Docs by LangChain home page![light logo](https://mintcdn.com/langchain-5e9cc07a/Xbr8HuVd9jPi6qTU/images/brand/langchain-docs-teal.svg?fit=max&auto=format&n=Xbr8HuVd9jPi6qTU&q=85&s=16111530672bf976cb54ef2143478342)![dark logo](https://mintcdn.com/langchain-5e9cc07a/Xbr8HuVd9jPi6qTU/images/brand/langchain-docs-lilac.svg?fit=max&auto=format&n=Xbr8HuVd9jPi6qTU&q=85&s=b70fb1a2208670492ef94aef14b680be)](/)
[github](https://github.com/langchain-ai)[x](https://x.com/LangChainAI)[linkedin](https://www.linkedin.com/company/langchain/)[youtube](https://www.youtube.com/@LangChain)
Resources
[Forum](https://forum.langchain.com/)[Changelog](https://changelog.langchain.com/)[LangChain Academy](https://academy.langchain.com/)[Trust Center](https://trust.langchain.com/)
Company
[About](https://langchain.com/about)[Careers](https://langchain.com/careers)[Blog](https://blog.langchain.com/)
[github](https://github.com/langchain-ai)[x](https://x.com/LangChainAI)[linkedin](https://www.linkedin.com/company/langchain/)[youtube](https://www.youtube.com/@LangChain)
[Powered by Mintlify](https://www.mintlify.com?utm_campaign=poweredBy&utm_medium=referral&utm_source=langchain-5e9cc07a)