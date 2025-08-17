from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="gemma3n:e4b")

chain = prompt | model

request = chain.invoke({"question": "What is LangChain?"})
request = model.invoke("where is Montaldo di mondovi?")
request = model.invoke("где находится  Montaldo di mondovi?")

print(request)