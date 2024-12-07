from openai import OpenAI
import os, csv
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("openai_api_key")

class GPT: 

    def __init__(self, model_name):
        self.model_name = model_name
        pass

    def write_message(self, prompt, system_prompt="", logging=False):

        client = OpenAI(api_key=api_key)

        completion = client.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        )

        result = completion.choices[0].message.content

        if logging:
            # Log prompt, result, timestamp, and model used
            log_file = 'LLM_logging.csv'
            file_exists = os.path.isfile(log_file)

            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)

                # Write header if file does not exist
                if not file_exists:
                    writer.writerow(["Timestamp", "Model Name", "Prompt", "Response"])

                # Write the log entry
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.model_name, prompt, result.replace('\n', ' ')])


        return result