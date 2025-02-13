[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/blanche)](https://pypi.org/project/blanche/)
[![Discord](https://img.shields.io/discord/1312234428444966924?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://discord.gg/atbh5s6bts)
[![PyPI Downloads](https://static.pepy.tech/badge/blanche)](https://pepy.tech/projects/blanche)

# Blanche 🌌

**[Blanche](https://www.blanche.cc/?ref=github) is a web browser for LLM agents.** It transforms the internet into an agent-friendly environment, turning websites into structured, navigable maps described in natural language. By using natural language commands, Blanche minimizes hallucinations, reduces token usage, and lowers costs and latency. It handles the browser complexity so your LLM policies can focus on what they do best: conversational reasoning and planning.

## A new paradigm for web agent navigation:

- Language-first web navigation, no DOM/HTML parsing required
- Treats the web as a structured, natural language action map
- Reinforcement learning style action space and controls

# Install

Requires Python 3.11+

```bash
pip install blanche
playwright install --with-deps chromium
```

# Config

Blanche uses language models to parse and structure web pages into a structured action space. To get started, you need to provide at least one API key for a supported language model provider. These keys can be configured in `.env` file and loaded into your environment;

```python
os.environ["OPENAI_API_KEY"] = "your-api-key"
# or any other provider(s) you have keys for
```

### Supported default providers

By default, Blanche supports the following providers:

- [Cerebras](https://cerebras.ai/inference) fastest, 60K tpm rate limit, wait-list keys
- [Anthropic](https://docs.anthropic.com/en/docs/api/api-reference) 40K tpm rate limit
- [OpenAI](https://platform.openai.com/docs/guides/chat/introduction) 30k tpm rate limit
- [Groq](https://console.groq.com/docs/api-keys) fast, 6K tpm rate limit

# Usage

As a reinforcement learning environment to get full navigation control;

```python
import os
from blanche.env import BlancheEnv

# setting fast language model provider keys
os.environ['ANTHROPIC_API_KEY'] = "your-api-key"

# Important: this should be run in an async context (e.g. notebook, asyncio, etc.)
# if you are running in a script, you should start `asyncio.run(main())`
async with BlancheEnv(headless=False) as env:
  # observe a webpage, and take a random action
  obs = await env.observe("https://www.google.com/travel/flights")
  obs = await env.step(obs.space.sample(role="link").id)
```

The observation object contains all you need about the current state of a page (url, screenshot, list of available actions, etc.);

```bash
> obs = env.observe("https://www.google.com/travel/flights")
> print(obs.space.markdown()) # list of available actions
```

```
# Flight Search
* I1: Enters departure location (departureLocation: str = "San Francisco")
* I3: Selects departure date (departureDate: date)
* I6: Selects trip type (tripType: str = "round-trip", allowed=["round-trip", "one-way", "multi-city"])
* B3: Search flights options with current filters

# Website Navigation
* B5: Opens Google apps menu
* L28: Navigates to Google homepage

# User Preferences
* B26: Open menu to change language settings
...
```

You can also scrape data from the page using the `scrape` function;
```python
...
async with BlancheEnv(headless=False) as env:
  ...
  obs = await env.scrape()
print(obs.data) # data extracted from the page (if any)
```

```
# Flight Search inputs
- Where from?: Paris
- Where to?: London
- Departure: Tue, Jan 14

# Flight Search Results
20 of 284 results returned.
They are ranked based on price and convenience

| Airline       | Departure  | Arrival  | Duration   | Stops     | Price |
|---------------|------------|----------|------------|-----------|-------|
| easyJet       | 10:15 AM   | 10:35 AM | 1 hr 20 min| Nonstop   | $62   |
| Air France    | 4:10 PM    | 4:35 PM  | 1 hr 25 min| Nonstop   | $120  |
```

Or alternatively, you can use Blanche conversationally with an LLM agent:

```bash
$ python examples/agent.py --goal "subscribe to blanche.cc newsletter with ap@agpinto.com"
```

🌌 Use Blanche as a backend environment for a web-based LLM agent. In this example, you integrate your own LLM policy, manage the interaction flow, handle errors, and define rewards, all while letting Blanche handle webpages parsing/understanding and browser interactions.

# API services

We offer managed cloud browser sessions with the following premium add-ons:

- **Authentication:** Built-in auth for secure workflows.
- **Caching:** Fast responses with intelligent caching.
- **Action Permissions:** Control over sensitive actions.

Request access to a set of API keys on [blanche.cc](https://www.blanche.cc/?ref=github)

Then integrate with the SDK;

```python
from blanche.sdk import BlancheClient
url = "https://www.google.com/flights"
with BlancheClient(api_key="your-api-key") as env:
    # Navigate to the page and observe its state
    obs = env.observe(url=url)
    # Interact with the page - type "Paris" into input field I1
    obs = env.step(action_id="I1", params="Paris")
    # Print the current state of the page
```



# Main features

- **Web Driver Support:** Compatible with any web driver. Defaults to Playwright.
- **LLM Integration:** Use any LLM as a policy engine with quick prompt tuning.
- **Multi-Step Actions**: Navigate and act across multiple steps.
- **Extensible:** Simple to integrate and customize.

# Advanced Config

### Using multiple keys

If you supply multiple keys in your `.env` file, Blanche uses a [llamux](https://github.com/andreakiro/llamux-llm-router) configuration to intelligently select the best model for each invocation. This approach helps avoid rate limits, optimize cost-performance balance, and enhance your experience. You can add more providers or adjust rate limits by modifying the [config file](blanche/llms/config/endpoints.csv)

# Contribute

Setup your local working environment;

```bash
poetry env use 3.11 && poetry shell
poetry install --with dev
poetry run playwright install
poetry run pre-commit install
```

Find an issue, fork, open a PR, and merge :)

# License

Blanche is released under the [Apache 2.0 license](LICENSE)
