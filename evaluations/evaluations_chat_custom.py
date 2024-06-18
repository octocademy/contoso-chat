import os
import sys
import json
import pandas as pd

from promptflow.core import AzureOpenAIModelConfiguration, Prompty

# Add the contoso_chat directory to the sys.path
sys.path.append(os.path.abspath('../contoso_chat'))

from chat_request import get_response

if __name__ == '__main__':
  
  # Initialize Azure OpenAI Connection
  model_config = AzureOpenAIModelConfiguration(
          azure_deployment="gpt-4",
          api_key=os.environ["AZURE_OPENAI_API_KEY"],
          api_version=os.environ["AZURE_OPENAI_API_VERSION"],
          azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
      )
  
  import pandas as pd

  data_path = "../data/data.jsonl"

  df = pd.read_json(data_path, lines=True)
  df.head()
  
  results = []

for index, row in df.iterrows():
    customerId = row['customerId']
    question = row['question']
    
    # Run contoso-chat/chat_request flow to get response
    response = get_response(customerId=customerId, question=question, chat_history=[])
    
    # Add results to list
    result = {
        'question': question,
        'context': response["context"],
        'answer': response["answer"]
    }
    results.append(result)
    
    # Save results to a JSONL file
    with open('result.jsonl', 'w') as file:
        for result in results:
            file.write(json.dumps(result) + '\n')
            
    # load apology evaluatorfrom prompty
    groundedness_eval = Prompty.load(source="prompty/groundedness.prompty", model={"configuration": model_config})
    fluency_eval = Prompty.load(source="prompty/fluency.prompty", model={"configuration": model_config})
    coherence_eval = Prompty.load(source="prompty/coherence.prompty", model={"configuration": model_config})
    relevance_eval = Prompty.load(source="prompty/relevance.prompty", model={"configuration": model_config})
    
    # Evaluate results from results file
    results_path = 'result.jsonl'
    results = []
    with open(results_path, 'r') as file:
        for line in file:
            results.append(json.loads(line))

    for result in results:
        question = result['question']
        context = result['context']
        answer = result['answer']
        
        groundedness_score = groundedness_eval(question=question, answer=answer, context=context)
        fluency_score = fluency_eval(question=question, answer=answer, context=context)
        coherence_score = coherence_eval(question=question, answer=answer, context=context)
        relevance_score = relevance_eval(question=question, answer=answer, context=context)
        
        result['groundedness'] = groundedness_score
        result['fluency'] = fluency_score
        result['coherence'] = coherence_score
        result['relevance'] = relevance_score

    # Save results to a JSONL file
    with open('result_evaluated.jsonl', 'w') as file:
        for result in results:
            file.write(json.dumps(result) + '\n')

    # Print results
    df = pd.read_json('result_evaluated.jsonl', lines=True)
    df.head()
    
    fmtresult = df.drop(['context', 'answer'], axis=1)
    
    headers = ["ID", "Question", "Relevance", "Fluency", "Coherence", "Groundedness"]
        
    fmtresult.to_markdown('result_evaluated.md', headers=headers)