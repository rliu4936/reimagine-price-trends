from Renderer import chart_dataset as cd
from Experimenter import cnn_experiment as ce 

exp_obj = ce.get_exp_obj_by_spec(
            20,
            5,
            country="CN",
            drop_prob=0.5
        )

exp_obj.generate_current_ensem_res("week", True)