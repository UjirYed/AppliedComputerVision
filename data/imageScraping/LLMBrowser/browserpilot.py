import sys
import os

#sys.path.append("browserpilot/browserpilot/agents/")

from gpt_selenium_agent import GPTSeleniumAgent

instructions = """
BEGIN_FUNCTION search_dining_halls
- Go to Google.com
- Find the element that says "Images".
- Click on this element.
- Find all textareas
- Find the first visible textarea
- Click on the first visible textarea.
- Type in "dining hall images" and press enter.
- Scroll to the bottom.
- Wait 3 seconds.
- Get the next fifty elements that have an id containing substring "dimg" and a height >= 100 and width >= 100. Avoid duplicate images.
- For each element, get the "src" attribute, and store to a list. 
- Scroll to the bottom and repeat getting the first fifty elements 3 times.

- Write this list to a file called images.txt
END_FUNCTION

RUN_FUNCTION search_dining_halls
- Wait for 5 seconds."""
#- print the list of src attributes.



if __name__ == "__main__":
    with open("queries.txt") as f:
        queries = f.read().splitlines() 
    
    for query in queries:
        print(query)
        search_text = "$REPLACEME$"

        replace_text = query

        with open(r'base_instructions.yaml', 'r') as file: 
            # Reading the content of the file 
            # using the read() function and storing 
            # them in a new variable 
            data = file.read() 
        
            # Searching and replacing the text 
            # using the replace() function 
            data = data.replace(search_text, replace_text)
            data = data.replace("$QUERYFILE$", query)
        
        # Opening our text file in write only 
        # mode to write the replaced content 
        with open(r'instructions2.yaml', 'w') as file: 
        
            # Writing the replaced data in our 
            # text file 
            file.write(data) 
    
        # Printing Text replaced 
        instructions = open("instructions2.yaml")
        print(type(instructions))
        agent = GPTSeleniumAgent(instructions = instructions,
                                chromedriver_path="chromedriver-mac-arm64/chromedriver",)


        agent.run()

#For each element, click on the element. Then, get the first <img> element on the page. Then, save the src of the img to a "sources" list. Then, go back to the previous page.