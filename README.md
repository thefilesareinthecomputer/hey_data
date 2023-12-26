VERSION

    hey_data 0.1.1
    

DEPENDENCIES

    macOS Sonoma 14

    python3.11

    brew install portaudio
    brew install flac

    pip install --upgrade pip
    pip install --upgrade virtualenv
    pip install --upgrade pytest
    pip install --upgrade wheel
    pip install --upgrade certifi
    pip install --upgrade setuptools

    requirements.txt

    This version utilizes the built-in macOS text to speech (TTS) engine, and will need slight modification on windows and linux with pyttsx3 or other TTS libraries.


USER DEFINED VARIABLES:

    within the .env file, optionally declare any of these variables (or others of your own) to extend tool functionality to the assistant:
    PROJECT_VENV_DIRECTORY=/Users/USERNAME/REPOSITORIES_FILDER/REPOSITORY/VENV
    USER_DOWNLOADS_FOLDER=/Users/USERNAME/Downloads
    WOLFRAM_APP_ID - Required to use the "robot, computation engine" function
    OPENAI_API_KEY - Required to use the "robot, call chatgpt" function
    GOOGLE_CLOUD_API_KEY
    GOOGLE_MAPS_API_KEY
    GOOGLE_GEMINI_API_KEY - Required to use the "robot, call gemini" function
    OPEN_WEATHER_API_KEY - Required to use the "robot, weather forecast" function
    USER_PREFERRED_LANGUAGE - 2 letter lowercase language code: USER_PREFERRED_LANGUAGE=en
    USER_PREFERRED_VOICE - Preferred voice name from built-in iOS TTS: USER_PREFERRED_VOICE=Daniel
    USER_PREFERRED_NAME - First name or nickname (conversational): USER_PREFERRED_NAME=Name
    USER_SELECTED_HOME_CITY - Full spelling of city: USER_SELECTED_HOME_CITY=City
    USER_SELECTED_HOME_COUNTY - Full spelling of county: USER_SELECTED_HOME_COUNTY=County
    USER_SELECTED_HOME_STATE - Full spelling of state / province: USER_SELECTED_HOME_STATE=State
    USER_SELECTED_HOME_COUNTRY - Country code in this format: USER_SELECTED_HOME_COUNTRY=ZZ
    USER_SELECTED_HOME_LAT - Floating point number in this format: USER_SELECTED_HOME_LAT=50.000000
    USER_SELECTED_HOME_LON - Floating point number in this format: USER_SELECTED_HOME_LON=-10.000000
    USER_SELECTED_PASSWORD - Add an optional passkey like this: USER_SELECTED_PASSWORD=password
    USER_SELECTED_TIMEZONE - User's local timezone in this format: USER_SELECTED_TIMEZONE=Country/City
    USER_CRYPTO - List of coins the user wants to track in this format: BTC,ETH,ADA,LTC
    USER_STOCK_WATCH_LIST - List of stocks the user wants to track in this format: USER_STOCK_WATCH_LIST=AAPL,GOOG,MSFT,VOO


APP OVERVIEW & NOTES:

    this is a voice activated ai assistent app designed to turn the user's laptop into a voice activated command center, note taker, question answerer, research assistant, information organizer/retriever, task automator, etc.
    when the app is running, it listens for user input and waits until it hears the activation word.
    the activation word, followed by one of the pre-determined hard-coded phrases, will trigger various functions based on the phrases. some will be quick one-off actions, and others will trigger sub-loops such as the 'robot, call gemini' command which will enter a stateful TTS<>STT chat loop with the gemini chatbot via the Google AI Studio SDK API.
    the user interacts with the app by speaking the activation word (a global constant variable) followed by predetermined phrases.
    
    The current operational phrases are:
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
                WIKI NOTES:
                the app crashes when wikipedia doesn't return a valid result. 
                the bot should list the next 3 closest results and ask the user if one of the 'next closest search results' is acceptable and if so, read it, and if not, then the bot should ask the user to rephrase the query.
                the bot recites the full wikipedia summary which can tend to be long. the user wants a way to interrupt the bot if necessary by saying "robot reset robot".
    - "robot, youtube video" to search youtube
    - "robot, computation engine" to interact with Wolfram|Alpha
                WOLFRAM ALPHA NOTES:
                functional, in UAT, clunky.
                the pods need to be summarized better - consolidated into a text to speech output that makes sense of the most relevant results returned from the wolfram alpha api in a concise but informative manner.
                once the contents of these pods have been aggregated into the answer variable, we need to summarize them before they are passed into the text to speech output.
    - "robot, weather forecast" to get a local weather forecast by day part
    - "robot, stock report" to open a dialogue about stocks (discounts, recommendations, yesterday, world, single)
    - "robot, call gemini" to interact with the Gemini chatbot (then say "robot, terminate chat" to exit back to the main chat loop)    
    - "robot, call chatgpt" to interact with chatgpt.
                currently not returning a successful response from the openai api due to quota limits but the account is fully paid and should be working. we need to debug this.
    - "robot, scientific research"
                not complete. we need to add the ability to search for health information from a list of trusted sources and summarize the results.
    - "robot, legal research"
                not complete. we need to add the ability to search for legal information from a list of trusted sources and summarize the results.


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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
