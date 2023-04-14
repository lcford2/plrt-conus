
## Notes from Sankar Meeting - 4/14/23

- Decompose components of experimental error on both axis (find what contibutes to NSE changes)
- Bar charts for parameterizations
- Make sure to make the discussion about what we can do moving forward

- Thoughts:
	- We are not necessarily dependent on the training set for a good model
		- The experiments show this
		- Clearly we can leave out entire groups of reservoirs, get different trees, and still expect nearly the same results in aggregate.
	- We should definitely check if the size of the training set matters
	- Also, since the PLRT can extract operational patterns for reservoirs in entire groups that are not included in training, it can most likely capture changes in operation by recalculating the residence time or using an effective storage capacity. 
	- This is really powerful because it means we are not beholden to only modeling past operations, but the flexibility provided by training a generalized model on MANY reservoirs could enable reliable prediction of future operations, even if those change for a particular reservoir.
