import streamlit as st
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_cohere import ChatCohere
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import tool
import schedule
import time
from datetime import datetime
from modulwa import output_str, df3
import pypdf

st.title("📈🔗 invest AI")
new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 20px;">Your own Financial advisor.</p>'
st.markdown(new_title, unsafe_allow_html=True)
llm = ChatCohere(cohere_api_key="api_key")

with st.sidebar:
    uploaded_file = st.file_uploader(label="Upload a file", type=['csv', 'pdf'])
    if uploaded_file is not None:
        fixed_file_name_csv = "file.csv"
        fixed_file_name_pdf = "file.pdf"
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        # To save file in the current directory:
        with open(fixed_file_name_csv, "wb") as file:
            file.write(bytes_data)

    Loader = CSVLoader(file_path='file.csv')
    data = Loader.load()


    prompt = (f"You are a highly experienced financial analyst, trader, and investor. Given historical stock data for a company in {data} which you should analyse very carefully , including Date, Open, High, Low, Close, Adjusted Close, and Volume, analyze the data thoroughly and provide a detailed report and recommendation for potential investment strategies. Follow these steps:\n"
          f"1. Data Overview: Summarize overall trends and significant patterns in the stock data, noting any notable highs, lows, and volume changes.\n"
          f"2. Technical Analysis: Identify primary trends over different time frames, determine key support and resistance levels, identify significant chart patterns, and calculate and interpret technical indicators such as Moving Averages, RSI, MACD, Bollinger Bands, and others.\n"
          f"3. Fundamental Analysis: Analyze the impact of recent earnings reports, news, and events on stock price movements; discuss valuation metrics such as P/E ratio and PEG ratio if available.\n"
          f"4. Sentiment Analysis: Evaluate market sentiment using volume and price action, summarize sentiment from recent news articles and social media discussions related to the company.\n"
          f"5. Investment Recommendation: Assess risk, provide strategies for different investment horizons (short-term, medium-term, long-term), and give clear, actionable advice on whether to buy, hold, or sell the stock.\n"
          f"6. Additional Insights: Offer any additional insights or considerations, such as macroeconomic factors or industry trends."
          f"7. Give a short answer . 1 line for 1 point . You can also give statistical example from the data .Total response should be less than 10 lines .")


    with st.form("csv_form"):
        insight = st.form_submit_button("Get Insights")
        if insight:
            csvr = llm.invoke(prompt)
            st.info(csvr.content)

    with st.form("data_chat"):
        query = st.text_area("Chat with data:")
        qsub = st.form_submit_button("Enter")
        prompt2 = f"You are provided {data} carefully observe this data, do your ananlysis and respond your answer as if you are expert trader and investor based on the query user will give. The user query is {query}. Be extremely short, extremely precise and exactly to the point to user query."
        if qsub:
            response = llm.invoke(prompt2)
            st.info(response.content)


def generate_response(text):
    tools = load_tools(["llm-math","wikipedia","google-serper"], llm=llm,serper_api_key="api_key")
    @tool
    def predict(text: str) -> str:
        """
        Returns the predicted value of the stock for next five days . This returns a string that contains predicted value \
        for stoch that has been calculated . Use this for any prediction queries and for any questions regarding stock price prediction \
        The input should always be an empty string
        """
        return output_str

    agent= initialize_agent(
        tools + [predict],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True)
    st.info(agent(text)['output'])

with st.form("my_form"):
    text = st.text_area("Enter text:", "Which stock to Invest?")
    submitted = st.form_submit_button("Submit")

    if submitted:
        generate_response(text)

with st.form("forecast"):
        fore = st.form_submit_button("Forecast")
        if fore:
            st.line_chart(df3)
