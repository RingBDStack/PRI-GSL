# squirrel
python main.py --beta=2 --dataset_name=squirrel --degree_ratio=0.5 --graph_learn_epsilon=0.3 --graph_learn_ratio=0.3 --learning_rate=0.01 --pri_loss_weight=1 --use_pri=1 --use_wave=1 --wave_weight=3

# chameleon
python main.py --beta=0.5 --dataset_name=chameleon --degree_ratio=0 --graph_learn_epsilon=0.7 --graph_learn_ratio=0.3 --learning_rate=0.01 --pri_loss_weight=0.6 --use_pri=1 --use_wave=1 --wave_weight=3

# citeseer
python main.py --beta=2 --dataset_name=citeseer --degree_ratio=5 --graph_learn_epsilon=0.5 --graph_learn_ratio=0.5 --learning_rate=0.01 --pri_loss_weight=1 --use_pri=1 --use_wave=1 --wave_weight=3

# cora
python main.py --beta=1 --dataset_name=cora --degree_ratio=1 --graph_learn_epsilon=0.3 --graph_learn_ratio=0.7 --learning_rate=0.01 --pri_loss_weight=0.1 --use_pri=1 --use_wave=1 --wave_weight=1

# photo
python main.py --beta=1 --dataset_name=photo --degree_ratio=5 --graph_learn_epsilon=0.5 --graph_learn_ratio=0.5 --learning_rate=0.001 --pri_loss_weight=2 --use_pri=1 --use_wave=1 --wave_weight=3






