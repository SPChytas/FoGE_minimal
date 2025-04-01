
LOG_LEVEL = 1

# level 0: nothing
# level 1: info
# level 2: todo
# level 3: debug

priority = {'info': 1,
            'debug': 3,
            'todo': 2,
            'unknown': 1}

keyword = {'info': '[INFO] ',
           'debug': '[DEBUG] ',
           'todo': '[TODO] ',
           'unknown': '[UNKNOWN] '}

def set_log(val):
    global LOG_LEVEL
    LOG_LEVEL = val 

def log(msg, msg_type='unknown'):
    global LOG_LEVEL
    global priority 
    global keyword

    if (msg_type not in priority):
        msg_type = 'unknown'

    if (priority[msg_type] <= LOG_LEVEL):
        print (keyword[msg_type] + msg)

