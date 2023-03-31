import numpy as np
from baseProblem import baseProblem

class RiverPollution(baseProblem):

    def __init__(
        self,
        name="RiverPollution",
        num_of_variables=2,
        num_of_objectives=4,
        num_of_constraints=0,
        upper_limits=1,
        lower_limits=0.3,
        ):
        super(RiverPollution, self).__init__(
            name,
            num_of_variables,
            num_of_objectives,
            num_of_constraints,
            upper_limits,
            lower_limits,
        )
        

    def objectives(self, decision_variables):
        self.f1= -1*(4.07 + 2.27 * decision_variables[0])

        self.f2= -1*(2.6
                + 0.03
                * decision_variables[0]
                + 0.02
                * decision_variables[1]
                + 0.01
                / (1.39 - decision_variables[0] ** 2)
                + 0.3
                / (1.39 - decision_variables[1] ** 2))

        self.f3= -1 * (8.21 - 0.71 / (1.09 - decision_variables[0] ** 2))

        self.f4= (0.96 * (1 / (1.09 - decision_variables[1] ** 2) - 1))


        self.obj_func = np.vstack([self.f1,self.f2,self.f3,self.f4])
        objc=np.zeros((self.num_of_objectives))
        objc[0]=self.obj_func[0][0]
        objc[1]=self.obj_func[1][0]
        objc[2]=self.obj_func[2][0]
        objc[3]=self.obj_func[3][0]

        return objc


# dv = [0.1, 0.2]
# mohsenProblem = RiverPollution("RiverPollution", 2, 4).objectives(dv)
# bv=0