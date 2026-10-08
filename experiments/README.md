We have many messages, but Discord only allows me to see my own messages. We have to backtranslate
(guess / simulate the message I responded to) using a cheap model, like Haiku 5.5. 

Before we backtranslate, we must address the elephant in the room: many of my messages are sent 
in chunks, not as one ocmplete message. To a model, this can be noisy. We can group related 
messages by time sent (i.e. it's highly probable that if i sent two messages within 30s of each 
other, they'd be part of the same message and should be counted as 1). 

Parameter to fill: Whats the best threshold? 1 min, 2 mins, 3 mins? That's what we're here to 
find out. 

