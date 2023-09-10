# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


#pip install -q transformers==4.4.2
#!pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html

import torch
import pandas as pd
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers import pipeline
from typing import Dict, Text, Any 

class ActionRunTransformers(Action):
    def name(self):
        return "action_run_transformers"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        # Extract the user's message to get the query
        user_message = tracker.latest_message.get('text')

        # Ensure that the user provided a query
        if not user_message:
            dispatcher.utter_message(text="Please provide a query.")
            return []

        # Your code goes here
        table = pd.read_csv(r"C:\Users\ntalashilkar\downloads\data.csv")
        table = table.astype(str)

        tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

        # Use the user's query
        query = user_message
        answer = tqa(table=table, query=query)["answer"]

        dispatcher.utter_message(text=f"The answer is: {answer}")

        return []
