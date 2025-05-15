import streamlit as st
import base64
import asyncio
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
import traceback
import base64
from client import send_request_to_server 

# --- Async call to MCP tool using PythonStdioTransport ---
async def call_tool_via_uv(user_input):
    transport = PythonStdioTransport("server.py")  # Ensure this matches the name/path of your server file
    async with Client(transport) as client:
        response = await client.call("retrieve_chunks", query=user_input)
        return response.get("result", "No result found.")

def generate_response(user_input: str) -> str:
    try:
        result = send_request_to_server(user_input)
        return result.get("result", "No result found.")
    except Exception as e:
        return f"Error while contacting server: {e}"

# --- Function to encode image to base64 ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Company logo and link ---
image_base64 = get_base64_image("codework_logo.jpg")  
homepage_url = "https://codework.ai"  

# --- Display clickable logo ---
st.markdown(
    f"""
    <div style='text-align: center; margin-bottom: 10px;'>
        <a href="{homepage_url}" target="_blank">
            <img src="data:image/png;base64,{image_base64}" alt="Codework Logo" style="width:200px;" />
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# --- App Title ---
st.title("CODEWORK PAL")

# --- Define static info for buttons ---
mission_vision_text = """
**Mission & Vision:**  
Our vision is to lead the way in delivering innovative and transformative technology solutions that reshape industries and empower businesses globally.
We strive to create a future where our software and services drive digital transformation, helping organizations optimize operations, boost efficiency, and unlock new growth opportunities.
Our mission is to leverage our expertise in software development to create meaningful solutions that benefit society while ensuring that our top talent shares in our organization's success.
At the core of our mission is a commitment to staying ahead of technological advancements, consistently pushing the limits of innovation to deliver cutting-edge solutions to our clients.
We are shaping the future of businesses, not just building technology.
"""

our_services_text = """
**Our Services:**  
- AI & Machine Learning Development  
- Software Development, Mobile Software Development and Custom Software Development  
- Cloud Computing 
- Web Designing
- IT Staff Augmentation
- Cybersecurity Services
- Penetration Testing
- DevOps Solutions  
Our AI solutions include expertise in Machine Learning and Predictive Analytics, Natural Language Processing and Generative AI with 
specializations in AI consulting, AI chatbots and AI automation.
"""

contact_text = """
**Contact Us:**  
Email: sales@codework.ai  
Phone: +91 75989 81500  
Website: [https://codework.ai](https://codework.ai)
"""

# --- Display buttons in a row ---
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Mission/Vision"):
        st.markdown(mission_vision_text)

with col2:
    if st.button("Our Services"):
        st.markdown(our_services_text)

with col3:
    if st.button("Contact"):
        st.markdown(contact_text)

# --- User Input and Bot Response ---
user_input = st.text_input("You:", placeholder="Hello! I’m here to assist you with all things Codework.")

if user_input:
    with st.spinner("Thinking..."):
        answer = generate_response(user_input)
        st.markdown(f"**Codework Pal:** {answer}")
