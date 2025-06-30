import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.eleven_labs import ElevenLabsTools
from agno.tools.firecrawl import FirecrawlTools
from agno.agent import RunResponse
from agno.utils.audio import write_audio_to_file
from agno.utils.log import logger
import streamlit as st
import uuid

st.set_page_config(page_title="Article to podcast agent")

st.title("Beginner Friendly End-to-End Agent")

st.sidebar.header("API keys")

openai_api_key = st.sidebar.text_input("OpenAI API key", type="password")
eleven_labs_api_key = st.sidebar.text_input("ElevenLabs API key", type="password")
firecrawl_api_key = st.sidebar.text_input("Firecrawl API key", type="password")

keys_provided=all([openai_api_key, eleven_labs_api_key, firecrawl_api_key])

url=st.text_input("Enter the URL of the sitr","")

generate_button=st.button("Generate podcast",disabled=not keys_provided or not url)

if not keys_provided:
    st.warning("Please enter all API keys")
    
if generate_button:
    if url.strip() == "":
        st.warning("Please enter a valid URL")
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["ELEVENLABS_API_KEY"] = eleven_labs_api_key 
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key

        with st.spinner("Processing... Scraping blog ,summarizing and generating podcast"):
            try:
                blog_to_podcast_agent=Agent(
                    name="blog to podcast agent",
                    agent_id="blog_to_podcast_agent",
                    model=OpenAIChat(id="gpt-4o-mini"),
                    tools=[
                        ElevenLabsTools(
                            voice_id="",
                            model_id="eleven_multilingual_v2",
                            target_directory="audio_generations"
                        ),
                        FirecrawlTools()
                    ],
                    description="You are an AI agent that can generate audio using ElevenLabs API.",
                    instructions=[
                        "When the user provide the bloig post URL."
                        "1. use FireCrawl tools to scrapr the blog content",
                        "2. create a concise summary of a blog content that is no more than 2000 characters long",
                        "3. the summary should capture the main points while engaging and conversational",
                        "4. use ElevenLabs tools to convert the summary to audio",
                        "ensure the summary is within 2000 character limit to avoid ElevenLabs API limit"
                    ],
                    markdown=True,
                    debug_mode=True
                )

                podcast: RunResponse= blog_to_podcast_agent.run(
                    f"Convert the blog content to podcast: {url}"
                )

                save_dir="audio_generations"
                os.makedirs(save_dir,exist_ok=True)

                if podcast.audio and len(podcast.audio)>0:
                    filename=f"{save_dir}/podcast_{uuid.uuid4()}.wav"
                    write_audio_to_file(
                        audio=podcast.audio[0].base64_audio,
                        filename=filename
                    )

                    st.success("Podcast generated successfully")
                    audio_bytes=open(filename,"rb").read()
                    st.audio(audio_bytes,format="audio/wav")
                    
                    st.download_button(
                        label="Download Podcast",
                        data=audio_bytes,
                        file_name=filename,
                        mime="audio/wav"
                    )

                else:    
                    st.error("Failed to generate podcast")

            except Exception as e:
                logger.error(f"Error generating podcast: {e}")
                st.error(f"Error generating podcast: {e}")

