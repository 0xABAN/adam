import json 

from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path 


def get_histogram(obj): 
    res = {
        '<5s': 0, 
        '5-30s': 0,
        '30s-2min': 0, 
        '2-10min': 0, 
        '10min-1hr': 0, 
        ">1hr": 0, 
    }
 
    for lines in obj.values(): 
        for i in range(1, len(lines)): 
            delta = lines[i][0] - lines[i-1][0]
            
            if delta < timedelta(seconds=5): 
                res['<5s'] += 1
            elif delta < timedelta(seconds=30): 
                res['5-30s'] += 1 
            elif delta < timedelta(minutes=2): 
                res['30s-2min'] += 1 
            elif delta < timedelta(minutes=10): 
                res['2-10min'] += 1 
            elif delta < timedelta(hours=1): 
                res['10min-1hr'] += 1 
            else: 
                res['>1hr'] += 1 
    
    return res     


def merge(obj, threshold): 
    res = defaultdict(list) 
    for channel, record in obj.items(): 
        stack = []
        for time, msg in record: 
            if stack and time - stack[-1][0] <= threshold: 
                _, prev_msg = stack.pop() 
                stack.append((time, prev_msg + " " + msg))
            else: 
                stack.append((time, msg))
        
        res[channel] = stack  

    return res 


def main(): 
    data_path = Path('data/discord/Messages/')
    message_path = 'messages.json'

    filtered = defaultdict(list) 
    counter = 0 
    
    # filter out pre-2022 messages
    for path in data_path.rglob(message_path):
        with open(path, 'r') as f: 
            data = json.load(f)
            data.reverse() # preserve chronological order
            
            for record in data: 
                time = datetime.fromisoformat(record['Timestamp'])
                msg = record['Contents'] 
                
                if time < datetime(2022, 1, 1): 
                    continue 
                    
                # if len(msg.split()) <= 1: 
                    # continue 
                 
                filtered[counter].append((time, msg));
                  
            counter += 1 
    
    T = 60 
    for _ in range(10):
        new = merge(filtered, timedelta(seconds=T))
        print(new)
        T += 60 
        

if __name__ == "__main__": 
    main() 
         
                

    

    
            

        
