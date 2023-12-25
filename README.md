VERSION

    hey_data 0.1.1
    
APP OVERVIEW & NOTES:

    this is a voice activated ai assistent app designed to turn the user's laptop into a voice activated command center.
    when the app is running, it listens for user input and waits until the input matches the activation word.
    the user interacts with the app by speaking the activation word followed by predetermined commands.
    
    
FUNCTIONAL ROBOT COMMANDS BY CATEGORY (these will likely become classes later):

GENERAL

    talk about yourself: "robot, talk about ('yourself' or 'what you can do')"
    functional, in UAT.


WEB SEARCH, RESEARCH, AND MEDIA

    google: "robot, google {query}"
    functional, in UAT.

    youtube video: "robot, youtube video"
    functional, in UAT.
    
    wikipedia summary: "robot, wiki research"
    the app crashes when wikipedia doesn't return a valid result. 
    the bot should list the next 3 closest results and ask the user if one of the 'next closest search results' is acceptable and if so, read it, and if not, then the bot should ask the user to rephrase the query.
    the bot recites the full wikipedia summary which can tend to be long. the user wants a way to interrupt the bot if necessary by saying "robot reset robot".

    computation engine: "robot, computation engine"
    functional, in UAT, clunky.
    the pods need to be summarized better - consolidated into a text to speech output that makes sense of the most relevant results returned from the wolfram alpha api in a concise but informative manner.
    once the contents of these pods have been aggregated into the answer variable, we need to summarize them before they are passed into the text to speech output.

    call chatgpt: "robot, chat gpt"
    currently not returning a successful response from the openai api due to quota limits but the account is fully paid and should be working. we need to debug this.

    weather forecast: "robot, weather forecast"
    functional, in UAT.
    
    health reasearch: "robot, health research"
    not complete. we need to add the ability to search for health information from a list of trusted sources and summarize the results.


FINANCE

    stock market report: "robot, stock report"
        "discounts" - stocks that are discounted from their 52 week high and on a growth trend for stocks in the user's watch list
        "recommendations" - stocks that are recommended to buy or sell for stocks in the user's watch list
        "daily" - % change vs yesterday for stocks in the user's watch list
        "world" - % change vs yesterday for the S&P 500, NASDAQ, and Dow Jones
        "single" - % change vs yesterday for a single stock
    functional, in UAT.
    
UTILITIES & SYSTEM COMMANDS

    click: "robot, click"
    clicks current cursor position.
    functional, in UAT.

    move cursor: "robot, {direction} {distance}"
    moves the cursor the specified number of pixels in the specified direction.
    functional, in UAT.

    translate: "robot, translate to {language}"
    functional, in UAT.
    this function is being interfered with because the bot is hearing its own output and mixing it with the user input.

    screenshot: "robot, screenshot"
    functional, in UAT.

    take notes: "robot, take notes"
    functional, in UAT.

    recall notes: "robot, recall notes"
    functional, in UAT.

    reset: "robot, reset robot" **DEPRECATING AND GOING TO RE-WRITE THIS LOGIC LATER
    for this to function, I think we need to route the text-to-speech output to another thread so that the app can continue to run and listen for user input while robot is speaking.
    when robot is asked to do research, the output can be unreasonably long at times and the user wants to be able to interrupt robot and reset the bot to the beginning of the chat loop to listen for another command.
    this feature is not working fully yet.
    robot currently outputs text to speech in chunks of words and then listens for input in between each chunk which is faulty because the user has to try to get input in in the very short pause between chunks which is not realistic.
    **DEPRECATING AND GOING TO RE-WRITE THIS LOGIC
    
    standby: "robot, standby mode" **DEPRECATING AND GOING TO RE-WRITE THIS LOGIC LATER
    functional, in UAT, but will be deprecated with the new logic for chatting and listening and the user being able to say 'robot, reset robot'.
    **DEPRECATING AND GOING TO RE-WRITE THIS LOGIC
 
    terminate program: "robot, terminate program"
    functional, in UAT.


    
BACKLOG (planned additions, improvements, and bug fixes):

    gain the ability to ingest knowledge from various media, interpret and summarize it, index it to a knowledge database (likely a graph database, maybe PostgreSQL), be able to query it in literal terms, and be able to converse about it with the user.
    stock analysis and advisor. Utilize the Yahoo Finance API for stock data.
    new voices for the speech interface. Investigate text-to-speech (TTS) libraries that offer a variety of voices. Python libraries like pyttsx3 or using third-party services like Google Cloud Text-to-Speech can provide diverse voice options.
    news report from tailored sources. Implement a feature to fetch news from APIs like NewsAPI, filtering content based on user preferences.
    communication (sms, google voice text, whatsapp text, signal text, email, etc.).
    add tqdm progress bars to long running tasks.
    consume knowledge from a youtube video playlist and then gain the ability to summarize the playlist, index it to a knowledge database (likely a graph databade), and converse about it with the user.
    translators: google translate for quick phrases, deepl for longer documents.
    click a link on the screen by name based on user speech.
    select a tab or window by description based on user speech (a description, such as 'top left of the screen, in back' or 'bottom right of the screen, in front', or 'minimized browser windows' or 'minimized VS code windows' or 'minimized images').
    find youtube videos by name or by subject and play them in the browser, play the audio, or summarize them.
    play spotify. Utilize the Spotify Web API for music playback controls.
    real estate analyzer. Utilize the Zillow API to fetch real estate data.
    system commands.
    meditation coach.
    restaurant reservations.
    google custom search engines.
    retrieval augmented generation from custom knowledge in a vector database or graph database with a semantic layer.
    add the ability to follow specific predefined user voice prompts to click windows, open links, type in fields, interact with UIs and apps, edit and crop, adjust settings like brightness and volume, etc. based on voice commands.
    add knowledge bases and retrieval augmented generation from custom knowledge in a vector database or graph database with a semantic layer.
    add the ability to talk to chatgpt, the api is not functioning. the assistant run is executing, but the api is not successfully returning a response. 
    vad for bot voice and background noice cancellation, or a way to filter out the bot's own voice from the user's input.
    chatgpt
        the chatgpt api code in this app is almost working but the chatgpt api is not fully working yet. 
        responses are not coming back from the model but the thread is running.
        we need to fix the non-functional response from the chatgpt api call and have the bot speak it back to the user. 
        we need to enter a stateful chat loop with chatgpt when chatgpt is called by the user. 
        the user must be able to exit the chat by saying "robot, end chat".
        i need debugging advice and direction please.    
    

COMPLETION LOG:

    2023-11-30 added google search, wikipedia search, and wolfram alpha query
    2023-12-01 note taking and note recall added
    2023-12-01 began threading the speech output to a separate thread so that in the future we can build the ability for the bot to listen for stop or change commands while speaking.
    2023-12-02 added more user details in the .env file to personalize the output in various functions.
    2023-12-03 speech queue and speech manager have been implemented to prevent the bot from trying to say multiple things at once.
    2023-12-03 wolfram alpha finction improved to consolidate various pods contained in the response from the api call rather than just the first pod.
    2023-12-03 screenshot function added.
    2023-12-03 verbal translation function added.
    2023-12-03 verbal translation function improved to output the translated text in the corresponding accent of the target language.
    2023-12-03 added the youtube search function.
    2023-12-04 finalized the spoken 4 day weather forecast function.
    2023-12-08 added the initial 'voice to move cursor' and 'voice to click functions' via pyautogui.
    2023-12-09 added stock report function.
    2023-12-09 added stock recommendation function.
    2023-12-10 moved all stock functions into a finance class.
    2023-12-20 began testing chat functionality with gemini rather than chatgpt with good success.
    2023-12-21 simplified the none handling in the main loop.
    2023-12-21 added the ability to enter a chat sub-loop with gemini chatbot by saying "robot, call gemini".
    2023-12-21 fixed a bug where the speech recognizer was retuurning 'None' vs None for unrecognized speech input.
    2023-12-22 installed auto py to exe and docker in anticipation of building a standalone app (tbd on containerization choice).
    
    
CURRENT SPRINT DETAILS:

    the speech timeout settings are still a bit clunky with room for improvement.
    currently, the bot is hearing its own output which is muddying the user input when the bot prompts the user for input.
    this is interfering with the ability to create a stateful chat loop with good conversational flow.
    the speech recognizer is combining the bot's speech and the user's speech into one message which is not correct.
    The i/o is currently working like this and needs to be fixed: 
        the user says "robot, translate to spanish", the bot says "Speak the phrase you want to translate.", the user says "this is the phrase." then the bot interprets and translates "Speak the phrase you want to translate. this is the phrase." into Spanish.
        this is also affecting other functions like the wikipedia summary function. 
        the bot also hears its own announcement when it sais "robot online", etc.
        we need the simplest solution possible to fix this problem.
    add the ability to conduct legal research with websites like casetext, lexisnexis, westlaw, docketbird, pacer, bloomberg law, bna, fastcase, bailii, barbri, bestlaw, case.law, case text, casecheck, case mine, case notebook,

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    The robot will listen for the activation word, then listen for a command.
    The phrases are:
    - "robot, reset robot" to reset the robot
    - "robot, standby mode" to put the robot on standby
    - "robot, terminate program" to end the program
    - "robot, talk about yourself" to hear the robot talk about itself
    - "robot, talk about what you can do" to hear the robot talk about its capabilities
    - "robot, screenshot" to take a screenshot of the screen
    - "robot, take notes" to take notes
    - "robot, recall notes" to recall notes
    - "robot, google search" to search google
    - "robot, click" to click the mouse
    - "robot, north {x pixels}" to move the mouse north
    - "robot, south {x pixels}" to move the mouse south
    - "robot, east {x pixels}" to move the mouse east
    - "robot, west {x pixels}" to move the mouse west
    - "robot, translate to {language}" to translate to a language
    - "robot, wiki research" to search wikipedia
    - "robot, youtube video" to search youtube
    - "robot, computation engine" to interact with Wolfram|Alpha
    - "robot, weather forecast" to get a local weather forecast by day part
    - "robot, stock report" to open a dialogue about stocks (discounts, recommendations, yesterday, world, single)
    - "robot, call gemini" to interact with the Gemini chatbot (then say "robot, terminate chat" to exit back to the main chat loop)
    i want to open a new restaurant in milan. i need to know which concepts are unsersaturated.
    recommend the 3 best ideas and explain why. think this through step by step before you answer.
    