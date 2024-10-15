# Sentiment analysis

Here you can train models on Rubtsova twit corpus with sentiment classes.

# Search best params:

1. Create model.py with param grid. You must add _'classifier__'_ to every parameter for right usage.
```python
import model_algorythm

param_grid = {
    'classifier__model': [1, 2, 3]
}

model = model_algorythm()

grid = get_full_pipline(model, param_grid, search='grid')
random = get_full_pipline(model, param_grid, search='random')
halv = get_full_pipline(model, param_grid, search='halving')pipline = get_full_pipline(model, param_grid)
```
There are 3 types of searching algorythms. 

2. Run 'python run.py model' to start search script.

3. You can also run train.py to train your model. There is parameter n to enlarge train set.

4. After all these tasks you may use gradio GUI to observe you models and get results. 
