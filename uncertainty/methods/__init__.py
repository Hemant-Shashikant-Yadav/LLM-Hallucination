"""
Objective 1: published uncertainty methods, as pure functions over stored generations.

`view.View` holds one question's greedy answer, valid samples and meaning-equivalence data;
each method maps a View to a score where **higher means more uncertain**, or None when the
method cannot run (no log-probs, too few samples). `registry.METHODS` lists them in report order.
"""
