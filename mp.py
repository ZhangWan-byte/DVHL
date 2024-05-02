import torch
import torch.multiprocessing as mp

def mytrain(x):
    x += 1
    print("Rank :{}, x: {}".format(mp.current_process().name, x))

x = 1

processes = [] 
for rank in range(4): 
    p = mp.Process( 
        target=mytrain, 
        args=(x,), 
        name=f"Process-{rank}"
    )
    p.start()
    processes.append(p)
    print(f"Started {p.name}") 
    
# Wait for all processes to finish
for p in processes:
    p.join()
    print(f"Finished {p.name}")