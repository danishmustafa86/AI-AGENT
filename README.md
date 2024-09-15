# Tech Jobs Recommender

This Streamlit app provides job recommendations based on job titles, job description summarization using LLaMA, translation of job descriptions, and job search functionality via Google Custom Search API. It also supports LinkedIn job searches and text-to-speech synthesis.

## Features

- **Job Recommendations**: Get job recommendations based on a selected job title.
- **Job Description Summarization**: Summarize job descriptions using LLaMA API.
- **Translation**: Translate job descriptions into multiple languages using Google Translator.
- **Job Search**: Search for jobs using Google Custom Search API and LinkedIn.
- **Search Linkedin Jobs**: You can alos search linkedin jobs just by giving your skill and locations.

## Installation

### 1. Clone the Repository

```bash


git clone https://github.com/danishmustafa86/AI-AGENT.git
cd tech-jobs-recommender

```

## Technologies
- Streamlit
- OpenAI
- LLaMA 3.1 8b
- Google Custom Search API
- CSV File

## Backend

The backend is built using Streamlit and provides endpoints for searching jobs according to the input provided by the user and also give the recommendations of different jobs by selecting the job with detail description of the selected jobs.Furthur more you cam also search of different latest articles and news about the available jobs in different locations just by giving the location. 


### Key Components

- **Jobs Recommendations**: Uses the `cvs` file to give recommendations when user select a specific job.
- **Job description**: Utilizes the LLaMA 3.1 model for giving detail description about the job, how it is important and many more points.
- **Translations**: Converts the job description in to any language so that everyone of those that are not able to understand english can read it in other languages.
- **Search Linkedin Jobs**: You can also search linkedin jobs just by giving the job name and it gives you the linkedin job search link where you can apply to the specific job that you input in the search bar.
- **Search Jobs articles**: You can also read different articles and news related to the skills, job and location you provide and you can read the any article you want to get more understanding about the available jobs.

## Setting Up the Environment

### Creating a Virtual Environment

Before running the application, it's recommended to create a virtual environment to isolate dependencies and avoid conflicts with other projects.

1. **Install `virtualenv`** (if not already installed):
    ```bash
    pip install virtualenv
    ```

2. **Create a virtual environment**:
    ```bash
    virtualenv venv
    ```
    This will create a directory named `venv` in your project directory, containing the isolated Python environment.

3. **Activate the virtual environment**:
    - **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    - **On macOS and Linux**:
        ```bash
        source venv/bin/activate
        ```

    After activation, your terminal prompt should change to indicate that you are now working within the virtual environment.

4. **Deactivate the virtual environment** (when done):
    ```bash
    deactivate
    ```

5. **Install backend dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

To run the Streamlit app:

1. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

### Notes

- If you get any error make sure to add the AI/ML Api key and Google Custom Search Api.
- Configure the API keys and endpoints according to your backend setup.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
