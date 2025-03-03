from openai import OpenAI, APIError

def call_api(prompt, temperature=0, max_tokens=300, max_retries=10, retry_interval=2):
    
    client = OpenAI(
        base_url="", #api url
        api_key="" #api key
    )

    inputs = prompt["prompt"]
    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=max_tokens,
                messages=inputs,
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except APIError as e:
            print(f"API error occurred: {e}. Retrying ({retries + 1}/{max_retries})...")
            retries += 1
            time.sleep(retry_interval)  # Fixed interval between retries
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying ({retries + 1}/{max_retries})...")
            retries += 1
            time.sleep(retry_interval)  # Fixed interval between retries

    raise Exception(f"Failed to summarize text after {max_retries} attempts.")
