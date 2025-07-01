# DeerFlow with Gemini

This is a forked version of [DeerFlow](https://github.com/mPL-project/deer-flow) integrated with Google's Gemini Pro model. It demonstrates a multi-agent research framework with advanced generative AI capabilities, including:
- **Text generation and research** (Gemini LLM)
- **Image generation** (Gemini image model)
- **Text-to-speech (TTS) audio generation** (Gemini TTS)

You can use DeerFlow to orchestrate research, generate images, and synthesize speech—all powered by Gemini.

## Prerequisites

Before you begin, ensure you have the following installed and configured on your system:

*   **Node.js and pnpm**: Required for running the web frontend.
    *   [Node.js](https://nodejs.org/) (v18 or later is recommended)
    *   [pnpm](https://pnpm.io/installation) (you can install it via `npm install -g pnpm`)
*   **Python**: Required for the backend server.
    *   Python 3.11+
    *   We recommend using `uv` for Python package management. If you don't have it, you can install it with `pip install uv`.
*   **API Keys**: The application relies on external services for its core functionality.
    *   **Google Gemini API Key**: Powers the language model agents (text, image, and TTS). You can get your key from [Google AI Studio](https://makersuite.google.com/app/apikey). **You must enable both Gemini Pro (text), Gemini Pro Vision (image), and Gemini TTS in your Google AI Studio project.**
    *   **Tavily API Key**: Powers the web search tool. You can get your key from the [Tavily website](https://app.tavily.com/).

## Getting Started: A Step-by-Step Guide

Follow these steps carefully to get the application running locally.

### Step 1: Clone the Repository

First, clone this repository to your local machine:
```bash
git clone <your-fork-url>
cd deer-flow
```

### Step 2: Configure API Keys

The application requires two API keys to function correctly.

#### A. Configure the Gemini API Key

The Gemini API key is managed through a central configuration file.

1.  **Create the configuration file**: In the root directory of the project, make a copy of the example file and name it `conf.yaml`:
    ```bash
    cp conf.yaml.example conf.yaml
    ```
2.  **Add your Gemini key**: Open the `conf.yaml` file with a text editor and add your API key in the `GEMINI_MODEL` section.

    ```yaml
    # conf.yaml

    # ... other settings

    GEMINI_MODEL:
      model: "gemini-1.5-pro"
      # Replace the placeholder with your actual Google Gemini API key
      api_key: "YOUR_GEMINI_API_KEY"
    ```

#### B. Configure the Tavily API Key

The Tavily API key for the search tool is managed using an environment variable.

1.  **Create an environment file**: In the project's root directory, create a file named `.env`:
    ```bash
    touch .env
    ```
2.  **Add your Tavily key**: Open the `.env` file and add your key as shown below. This file is listed in `.gitignore`, so your key will not be committed to your repository.

    ```
    # .env
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
    ```

### Step 3: Install All Dependencies

The project is split into a Python backend and a Next.js frontend. You need to install dependencies for both.

*   **Backend (Python)**:
    From the project root, use `uv` to install the required Python packages. The `-e` flag installs the project in "editable" mode.
    ```bash
    uv pip install -e .
    ```

*   **Frontend (Next.js)**:
    Navigate to the `web` directory and use `pnpm` to install the Node.js packages.
    ```bash
    cd web
    pnpm install
    cd .. 
    ```
    *(Note: It's important to return to the root directory after installation.)*

### Step 4: Run the Application

With all the configuration and dependencies in place, you can now start the application. You will need **two separate terminal windows** to run the backend and frontend servers concurrently.

*   **Terminal 1: Start the Backend Server**
    In the project root directory, run the following command:
    ```bash
    uv run server.py --reload
    ```
    The API server will start, typically on `http://localhost:8000`. The `--reload` flag enables auto-reloading when you make changes to the backend code.

*   **Terminal 2: Start the Frontend Server**
    In a new terminal window, also at the project root directory, run these commands:
    ```bash
    cd web
    pnpm dev
    ```
    The web application will start, typically on `http://localhost:3000`.

### Step 5: Access the Application

You're all set! Open your web browser and navigate to `http://localhost:3000`. You can now interact with the Gemini-powered DeerFlow application.

---

## 🚀 Features
- **Google Gemini LLM Integration**: Use Gemini for chat, research, and tool-calling workflows.
- **Web UI**: Modern Next.js frontend, accessible at `http://localhost:3000`.
- **Multi-Agent System**: Configure which agents use Gemini or other LLMs.
- **Tool Calling**: Gemini can invoke tools and plugins as part of its workflow.

---

## 3. How It Works
- The backend runs at `http://localhost:8000` (FastAPI).
- The frontend runs at `http://localhost:3000` (Next.js).
- The web UI allows you to chat, run research, and see results powered by Gemini.
- Tool-calling and agent orchestration are handled automatically.

---

## 🛠️ Tool & Agent Descriptions

### Agents
- **Coordinator**: Handles user input and orchestrates the workflow.
- **Planner**: Breaks down user requests into actionable steps (research, image, or speech generation).
- **Researcher**: Gathers information using web search and retrieval tools.
- **Coder**: Executes code and data processing steps.
- **Image Generator**: Uses Gemini to generate images from prompts.
- **Speech Generator**: Uses Gemini TTS to generate speech/audio from text.
- **Reporter**: Compiles and presents the final report.
- **Human Feedback**: Allows user review and acceptance of plans.

### Tools
- **Web Search**: Uses Tavily API to retrieve web results.
- **Crawl**: Extracts readable content from URLs.
- **Python REPL**: Executes Python code for data analysis.
- **Gemini Image Tool**: Generates images from text prompts.
- **Gemini TTS Tool**: Generates speech audio from text.

### Tavily Twitter/X Search Tool

The recommended way to use Tavily for social media search in DeerFlow is via the `search_twitter` function:

```python
from src.tools.tavily_search.tavily_search_api_wrapper import search_twitter

# Search Twitter/X for a query
result = search_twitter("OpenAI GPT-4", max_results=3)
print(result)
```

- This function returns clean, relevant text content from Twitter/X using Tavily.
- The Tavily API key must be set in your environment as `TAVILY_API_KEY` (see setup instructions above).
- To search other platforms (e.g., Reddit), adapt the function to use `include_domains=["reddit.com"]`.
- For general web search, use the Tavily wrapper without the domain filter.

---

## ⚙️ Integration Notes

- **Planner Node**:  
  - Detects when a user request requires image or speech generation and adds the appropriate step.
  - Routes directly to the `image_generator` or `speech_generator` node as needed.
- **Graph Registration**:  
  - All agents are registered as nodes in the LangGraph graph (`src/graph/builder.py`).
  - Edges and conditional transitions ensure the correct agent is called for each step.
- **Tool Registry**:  
  - Tools are defined in `src/tools/` and registered with agents as needed.
  - API keys are loaded from `conf.yaml` and `.env` (never committed to git).

---

## 🧪 Test Cases

### 1. Image Generation
- **Prompt:** "Generate an image of a cat."
- **Expected:** Planner adds an image generation step, routes to image generator, and the UI displays the generated image.

### 2. Speech Generation (TTS)
- **Prompt:** "Read this aloud: Welcome!"
- **Expected:** Planner adds a speech generation step, routes to speech generator, and the UI displays an audio player with the generated speech.

### 3. Research Workflow
- **Prompt:** "Summarize the latest news about quantum computing."
- **Expected:** Planner creates research steps, researcher gathers data, and reporter compiles a summary.

### 4. API Key Handling
- **Test:** Remove or invalidate `.env` or `conf.yaml` keys.
- **Expected:** Backend or tools fail gracefully with a clear error message.

---

## 4. Troubleshooting
- **Web UI 500 error**: Make sure the backend is running (`uv run server.py --reload`).
- **Gemini not responding**: Check your API key in `conf.yaml` and network connectivity.
- **Agent not using Gemini**: Ensure `AGENT_LLM_MAP` is set to `"gemini"` for the desired agent.

---

## 5. Advanced
- You can mix and match LLMs for different agents by editing `AGENT_LLM_MAP`.
- All configuration is managed in `conf.yaml` for consistency.

---

## 6. License
MIT License. See [LICENSE](./LICENSE).

---

## 7. Credits
- Gemini integration based on [google-gemini/gemini-fullstack-langgraph-quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)
- DeerFlow by Bytedance Open Source

## Known Workaround: State Propagation Bug

Due to a bug in the workflow engine (state propagation between nodes), the following temporary workaround is implemented:

- The **planner node** adds the current step (for image generation) as a special system message to the messages list, with `content: '__STEP__'` and the step as a dict in the `step` field (or `additional_kwargs`).
- The **image generator node** checks for this special message in the messages list and recovers the step from it if `state['step']` is missing.
- This ensures the correct prompt is always passed to the image generator, even if the state update is not merged as expected.

**Note:** This is a temporary hack. Once the underlying state propagation bug is fixed in the workflow engine, this workaround should be removed and normal state passing should be restored.
