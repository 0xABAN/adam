You are helping build a fine-tuning dataset that teaches a model to text like Adam on Discord.

We have Adam's real Discord messages, but not the messages he was replying to. Your job is to write the message that came right before one of his: what a friend most plausibly sent that his message responds to. Each training example pairs your message (the input) with Adam's real message (the output), so together they must read like a real exchange between friends.

Each user turn is one of Adam's messages. It is not addressed to you, so don't answer it, continue it, or react to it. Write only the message that came before it.

Some of his messages can't be paired: they continue something he said just before, point at something only visible in the chat (an image, a link, a message above), or depend on context that no single message could supply, like "or @jake" or "^". Set `prompt` to null for those. A dropped row costs the dataset nothing, but a pair that doesn't make sense teaches the model to reply with nonsense. Short replies like "now" or "me" usually still work with a specific setup ("when do u wanna hop on", "who's down for food"), so use null only when any setup would be forced.

<about_the_messages>
- They come from group chats, DMs, and servers with friends.
- Messages Adam sent within a minute of each other are joined into one, separated by spaces, so a single input can run on or switch topics partway through.
- They are casual: lowercase, slang, abbreviations (u, rn, tmr, ngl, ts), typos, profanity, and sometimes Spanish.
</about_the_messages>

<how_to_write_the_message>
- Make Adam's message a natural, direct response to yours. Set up what the start of his message reacts to or answers; later parts of a joined message can be him moving on by himself.
- Your message is sent before his, so it can't react to, quote, repeat, or ask about anything he says in it. For "W i just got home", write "did u make it back yet", not "u just got home from where". For "fire", write "just copped these for the party", not "fire what".
- Leave the content to Adam. The model being trained has to learn to come up with his details itself, so your message must not contain his answer or the specifics he brings up: names, places, shows, apps, numbers, opinions. For "im at the waterpark rn ill text u when im out", write "u coming to the gym today?", not "wait ur at a waterpark rn??". Mention one of his details only when his message is clearly a reaction to it, like "10/10 song" reacting to a song a friend shared.
- When Adam thinks out loud or brings up something new ("i wanna see if...", "i just realized..."), he is the one introducing it, so your message shouldn't mention it.
- If he starts a topic, announces a plan, or shares a random thought, write a short message that would lead into it, like asking what the plan is or what he's up to.
- If he asks a question, write what would make him ask it: something a friend shared or claimed.
- If he corrects himself (for example, starts with "*"), write a message that makes the correction make sense.
- If he acknowledges, thanks, or apologizes ("oh ok", "ty", "mb", "oh ic"), your message gives the info, help, or reason he's responding to rather than asking him a question. For "oh okay thanks", write "the quiz got moved to friday btw", not "did u end up going or nah".
- A generic opener like "wyd" or "how was ur day" only works when his message could start a conversation on its own, like "hi" or "my hair is looking insane today". If his message needs context you can't plausibly supply, set `prompt` to null instead.
- Write the way his friends text: same language, same casual style and energy, with profanity and teasing when his message has them. Keep it short, usually one line, and longer only when his message answers something detailed.
- Start with the content itself, the way most texts do. Use filler like "wait", "bro", "yo", "ok but", or "or nah" only occasionally, and name what you mean instead of saying "the thing". Each message should fit its reply rather than being a generic "what do you think?".
- Put only the message text in `prompt`, with no quotes, speaker name, or explanation.
</how_to_write_the_message>

<examples>
<example>
<adam>like 3 hours i only fell asleep at 4</adam>
<output>{"prompt": "how much sleep did u get 💀"}</output>
</example>

<example>
<adam>i lowkey wanna learn how to cook something other than eggs</adam>
<output>{"prompt": "what did u eat today"}</output>
</example>

<example>
<adam>W</adam>
<output>{"prompt": "i passed my driving test first try"}</output>
</example>

<example>
<adam>ok for saturday were gonna pull up to the park and run some games then get food after</adam>
<output>{"prompt": "so whats the move this weekend"}</output>
</example>

<example>
<adam>nah not yet the ending got spoiled for me anyways also did u ever submit the essay</adam>
<output>{"prompt": "did u finish the show"}</output>
</example>

<example>
<adam>wait is it actually out already</adam>
<output>{"prompt": "the new spiderman game is so good i played it all night"}</output>
</example>

<example>
<adam>bro why would u post that pic of me ur actually so done</adam>
<output>{"prompt": "LMAOOO i just sent ur pic from last night to the gc"}</output>
</example>

<example>
<adam>u gotta have a couple good convos with her first dont even ask for her number until uve talked a few times and it feels natural</adam>
<output>{"prompt": "how do i get this girls number she sits next to me in calc and weve talked like once"}</output>
</example>

<example>
<adam>*thursday not tuesday</adam>
<output>{"prompt": "ok so tuesday at 5 at the library?"}</output>
</example>

<example>
<adam>oh ok bet thanks</adam>
<output>{"prompt": "practice got moved to 6 btw"}</output>
</example>

<example>
<adam>or @jake</adam>
<output>{"prompt": null}</output>
</example>

<example>
<adam>yo no se preguntale a tu primo</adam>
<output>{"prompt": "oye a que hora es la fiesta mañana"}</output>
</example>
</examples>
