import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set upi the streamlit app
st.set_page_config(page_title= "Text to Math Problem Solver And Data search Assistant", page_icon="📚")
st.title("📚 Text To Math Problem Solver Using Google Gemma 2")

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key = groq_api_key)

#Initializing the tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    description="Search Wikipedia for various information",
    func= wikipedia_wrapper.run
)

## Initializing the math tool
math_chain = LLMMathChain.from_llm(llm= llm)
calculator = Tool(
    name = "Calculator",
    func = math_chain.run,
    description = "Solve math problems using Google Gemma 2"
    
)

prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)
chain = LLMChain(llm= llm, prompt= prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    description="A tool for answering the logic base and reasoning questions",
    func=chain.run
)

## initialize the agents

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a MAth chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## LEts start the interaction
question=st.chat_input("Enter youe question:")

# if st.button("find my answer"):
if question:
    with st.spinner("Generate response.."):
        st.session_state.messages.append({"role":"user","content":question})
        st.chat_message("user").write(question)

        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb]
                                        )
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write('### Response:')
        st.success(response)

else:
    st.warning("Please enter the question")



