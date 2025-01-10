from datetime import datetime

def save_markdown(task_output):
    # Verifique se task_output.result é uma string
    result_content = task_output.result if isinstance(task_output.result, str) else str(task_output.result)

    # Get today's date in the format YYYY-MM-DD
    today_date = datetime.now().strftime('%Y-%m-%d')
    # Set the filename with today's date
    filename = f"{today_date}.md"
    # Write the task output to the markdown file
    with open(filename, 'w') as file:
        file.write(result_content)
    print(f"Newsletter saved as {filename}")
