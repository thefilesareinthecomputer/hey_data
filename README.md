## VERSION

    hey_data 0.2.3
    

## LICENSE

### Open Source License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE.txt file for details. This open-source license is primarily for individual researchers, academic institutions, and non-commercial use. Contributions to the open-source version fall under the same GPLv3 licensing terms.

### Commercial License

For commercial use of this project, a separate commercial license is required. This includes use in a commercial entity or for commercial purposes. Please contact us at https://github.com/thefilesareinthecomputer for more information about obtaining a commercial license.


## DEPENDENCIES

    macOS Sonoma 14

    python3.11

    brew install portaudio
    brew install flac

    Java 17 for the neo4j graph database
    https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html

    neo4j community edition, locally hosted self managed graph database (place in the project root directory after unzipping)
    https://neo4j.com/deployment-center/?ref=subscription#community

    if you prefer to run the bot fully offline you'll need a speech to text model instead of the google api which is the SpeechRecognition package's default. you can also pass your own Google Cloud API key to the speech_recognition Recognizer if you want further control over your parsed data. if you opt to run it offline, ensure the pocketsphinx model is present in the speech_recognition directory once installed. if not, download and unzip it, then store it in the speech_recognition package directory.
    https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/US%20English/

    pip install --upgrade pip
    pip install --upgrade virtualenv
    pip install --upgrade pytest
    pip install --upgrade wheel
    pip install --upgrade certifi
    pip install --upgrade setuptools

    requirements.txt

    This version utilizes the built-in macOS text to speech (TTS) engine for the bot's voice, and will need slight modification on windows and linux with pyttsx3 or other TTS libraries.


## USER DEFINED VARIABLES:

    within the .env file, optionally declare any of these variables (or others of your own) to extend tool functionality to the assistant:
    PROJECT_VENV_DIRECTORY=
    PROJECT_ROOT_DIRECTORY=
    USER_DOWNLOADS_FOLDER=

    GOOGLE_CLOUD_API_KEY=
    GOOGLE_CLOUD_PROJECT_NUMBER=
    GOOGLE_CLOUD_PROJECT_ID=
    GOOGLE_MAPS_API_KEY=
    GOOGLE_GEMINI_API_KEY=
    GOOGLE_API_KEY=
    GOOGLE_DOCUMENTATION_SEARCH_ENGINE_ID=
    OPEN_WEATHER_API_KEY=
    WOLFRAM_APP_ID=
    OPENAI_API_KEY=
    ACTIVATION_WORD=
    PASSWORD=
    EXIT_WORDS=exit,quit,cancel,stop,break,terminate,end
    USER_PREFERRED_LANGUAGE=en
    USER_PREFERRED_VOICE=Oliver
    USER_PREFERRED_NAME=
    USER_SELECTED_HOME_CITY=
    USER_SELECTED_HOME_COUNTY=
    USER_SELECTED_HOME_STATE=
    USER_SELECTED_HOME_COUNTRY=
    USER_SELECTED_HOME_LAT=
    USER_SELECTED_HOME_LON=
    USER_SELECTED_TIMEZONE=

    JAVA_HOME=
    NEO4J_URI=
    NEO4J_USER=
    NEO4J_PASSWORD=
    NEO4J_DATABASE=
    NEO4J_PATH=

    USER_CRYPTO=
    USER_STOCK_WATCH_LIST=


## APP OVERVIEW & NOTES:

    this is a voice activated ai assistent app designed to turn the user's laptop into a voice activated command center, note taker, question answerer, research assistant, information organizer/retriever, task automator, etc.
    when the app is running, it listens for user input and waits until it hears the activation word.
    the activation word, followed by one of the pre-determined hard-coded phrases, will trigger various functions based on the phrases. some will be quick one-off actions, and others will trigger sub-loops such as the 'robot, call gemini' command which will enter a stateful TTS<>STT chat loop with the gemini chatbot via the Google AI Studio SDK API.
    the user interacts with the app by speaking the activation word (a global constant variable) followed by predetermined phrases.
    
    The current operational phrases are:
    - "robot, terminate program" to end the program
    - "robot, screenshot" to take a screenshot of the screen
    - "robot, take notes" to take notes
    - "robot, recall notes" to recall notes
    - "robot, google search" to search google
    - "robot, mouse click" to click the mouse
    - "robot, mouse {direction} {distance}" to move the mouse x pixels in y direction
    - "robot, translate to {language}" to translate to a language
    - "robot, wiki research" to search wikipedia
                WIKI NOTES:
                the app crashes when wikipedia doesn't return a valid result and needs error handling. 
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
    - "robot, custom search engine" to interact with the custom search engine.


## BACKLOG (planned additions, improvements, and bug fixes):

    gain the ability to ingest knowledge from various media, interpret and summarize it, index it to a knowledge database (likely a graph database, maybe PostgreSQL), be able to query it in literal terms, and be able to converse about it with the user.
    new voices for the speech interface. Investigate text-to-speech (TTS) libraries that offer a variety of voices. Python libraries like pyttsx3 or using third-party services like Google Cloud Text-to-Speech can provide diverse voice options.
    news report from tailored sources. Implement a feature to fetch news from APIs like NewsAPI, filtering content based on user preferences.
    communication (sms, google voice text, whatsapp text, signal text, email, etc.).
    add tqdm progress bars to long running tasks.
    consume knowledge from a youtube video playlist and then gain the ability to summarize the playlist, index it to a knowledge database (likely a graph database), and converse about it with the user.
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
    vad (voice activity detection) for bot voice and background noice cancellation, or a way to filter out the bot's own voice from the user's input.
    chatgpt
        the chatgpt api code in this app is almost working but the chatgpt api is not fully working yet. 
        responses are not coming back from the model but the thread is running.
        we need to fix the non-functional response from the chatgpt api call and have the bot speak it back to the user. 
        we need to enter a stateful chat loop with chatgpt when chatgpt is called by the user. 
        the user must be able to exit the chat by saying "robot, end chat".
        i need debugging advice and direction please.    
    integrate it with the phomemo printer to print notes.
    add the ability to conduct legal research with websites like casetext, lexisnexis, westlaw, docketbird, pacer, bloomberg law, bna, fastcase, bestlaw, case text, casecheck, case notebook.
    add oauth2 for authentication to google cloud API services with pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib.
    unit tests.
    input validations for each function.
    performance optimization.
    code profiling to identify bottlenecks.
    additional secutiry measures - encryption, access control, input sanitation, etc.
    to modularize the "chatbot tools initial conversation" we can implement args into the tools and then have the "inquiry" function get user input for each arg in a loop - for arg in args: arg = input("what is your value for " + arg + "?") - then we can pass the args into the tool function.



## CURRENT SPRINT DETAILS:

    the speech timeout settings are still a bit clunky with room for improvement.
    currently, the bot is hearing its own output which is muddying the user input when the bot prompts the user for input.
    this is interfering with the ability to create a lifelike stateful chat loop with good conversational flow.
    the speech recognizer is combining the bot's speech and the user's speech into one message which is not correct.
    The i/o is currently working like this and needs to be fixed: 
        the user says "robot, translate to spanish", the bot says "Speak the phrase you want to translate.", the user says "this is the phrase." then the bot interprets and translates "Speak the phrase you want to translate. this is the phrase." into Spanish.
        this is also affecting other functions like the wikipedia summary function. 
        the bot also hears its own announcement when it sais "robot online", etc.
        we need the simplest solution possible to fix this problem.
    planning to build the ui in flet .
    implement cachching for speed increase.
    add functionality to work with the ne04j graph database.


## COMPLETION LOG / COMMIT MESSAGES:

    0.1.1 - 2023-11-30 added google search, wikipedia search, and wolfram alpha query
    0.1.1 - 2023-12-01 note taking and note recall added
    0.1.1 - 2023-12-01 moved speech output to a separate thread so that in the future the bot can listen while speaking.
    0.1.1 - 2023-12-02 added more user details in the .env file to personalize the output in various functions.
    0.1.1 - 2023-12-03 speech queue and speech manager have been implemented to prevent the bot from trying to say multiple things at once.
    0.1.1 - 2023-12-03 wolfram alpha function improved to consolidate specified pods from the api call rather than just the first pod.
    0.1.1 - 2023-12-03 screenshot function added.
    0.1.1 - 2023-12-03 verbal translation function added.
    0.1.1 - 2023-12-03 verbal translation function improved with native accents for translated speech.
    0.1.1 - 2023-12-03 added the youtube search function.
    0.1.1 - 2023-12-04 finalized the spoken 4 day weather forecast function.
    0.1.1 - 2023-12-08 added the initial 'voice to move cursor' and 'voice to click functions' via pyautogui.
    0.1.1 - 2023-12-09 added stock report function.
    0.1.1 - 2023-12-09 added stock recommendation function.
    0.1.1 - 2023-12-10 moved all stock functions into a finance class.
    0.1.1 - 2023-12-20 began testing chat functionality with gemini rather than chatgpt with good success.
    0.1.1 - 2023-12-21 simplified the none handling in the main loop.
    0.1.1 - 2023-12-21 added the ability to enter a chat sub-loop with gemini chatbot by saying "robot, call gemini".
    0.1.1 - 2023-12-21 fixed a bug where the speech recognizer was retuurning 'None' vs None for unrecognized speech input.
    0.1.1 - 2023-12-22 installed auto py to exe and docker in anticipation of building a standalone app (tbd on containerization choice).
    0.1.1 - 2023-12-25 moved the activation word from hard-coded 'robot' into a user-defined variable in the .env file.
    0.1.1 - 2023-12-30 downloaded new voices for the speech interface.
    0.1.2 - 2024-01-01 removed the redundant second parsing attempt in the speech parsing function and simplified the error handling there.
    0.1.2 - 2024-01-01 removed the obsolete standby and reset code blocks to make space for a better future reset feature.
    0.1.2 - 2024-01-01 moved the mouse movement and clicking controls into a more streamlined function.
    0.1.2 - 2024-01-01 added verbal password check when the app runs.
    0.1.2 - 2024-01-01 moved the gemini chat loop into its own function.
    0.1.2 - 2024-01-01 integrated a google custom search engine and LLM search assistant agent with brief analysis of results.
    0.1.2 - 2024-01-02 added a prompt template into the gemini chat initialization asking for good output for a tts app.
    0.1.2 - 2024-01-02 added some additional prompt template steps into the search assistant chatbot.
    0.2.1 - 2024-01-03 re-built the neural nine chatbot that uses intents.json and modernized the tensorflow imports in the training module.
    0.2.1 - 2024-01-04 implemented function calling and stt audio recognition and tts bot output for the neural network based chatbot.
    0.2.1 - 2024-01-04 implemented a function that generates an intent json object for any message interaction the bot doesn't recognize.
    0.2.1 - 2024-01-04 imported and called the chatbot_training module at the top of the chatbot_app module.
    0.2.1 - 2024-01-05 improved mouse control function (responds to up down left right + north south east west) after migrating to v 0.2.1.
    0.2.1 - 2024-01-05 added a function to run diagnostics on the codebase with inspect and then call in the llm as a pair programmer copilot.
    0.2.2 - 2024-01-05 trimmed out commented old functions and migrated more tool methods from 0.1.2 to 0.2.2.
    0.2.2 - 2024-01-05 added a function for the llm to simply read in the codebase and stand by as a pair programmer.
    0.2.2 - 2024-01-05 added more thorough docstrings to all classes and methods.
    0.2.2 - 2024-01-06 added a shared class dictionary called data_store to the ChatBotTools class which stores the data rendered by certain chatbot tools like wikipedia for later access by the LLM agent.
    0.2.2 - 2024-01-07 migrated the remaining functions from 0.1.2 to 0.2.2.
    0.2.2 - implemented a neo4j graph database with the intention of turning this into the bot's full knowledge base.
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
