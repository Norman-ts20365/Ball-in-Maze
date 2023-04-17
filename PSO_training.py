'''
PSO Training Class for Fuzzy Logic Motion Controller
Oscar Dilley, April 2023
Univeristy of Bristol
Electrical and Electronic Engineering Year 3
'''
# Pytest functionality
import pytest
import csv
from random import randint, uniform, random
import numpy as np
import pandas as pd


class PsoTraining():
    '''
    This class is responsible for implementing a particle swarm optimisation heuristic to optimise 12 parameters for a fuzzy logic control
    system.
    '''
    def __init__(self):

        self.conf_path = r"C:\Users\oscar\OneDrive\Documents\UNI_work\Year_3\Group_Project\PSO_Training\config.csv"
        with open(self.conf_path, 'r') as config_file:
            conf_reader = csv.DictReader(config_file)
            # determine population, inertial weight, C1, C2, gbest and round number:
            config_values = {}
            for i in list(conf_reader):
                i = list(i.values())
                config_values.update({i[0]:i[1]})
            # Fixed values for algorithm
            self.population = int(float(config_values.pop('Population')))
            self.W = float(config_values.pop('Intertial_Weight'))
            self.c1 = float(config_values.pop('Cognitive_Constant'))
            self.c2 = float(config_values.pop('Social_Constant'))
            # Progress tracking
            self.round = int(float(config_values.pop('Round')))
            self.index = int(float(config_values.pop('Index')))
            # Current round start and target coords
            self.xStart = int(float(config_values.pop('X_Start')))
            self.yStart = int(float(config_values.pop('Y_Start')))
            self.xTarget = int(float(config_values.pop('X_Target')))
            self.yTarget = int(float(config_values.pop('Y_Target')))
            # Best case parameters
            self.bestCase = float(config_values.pop('Global_Best'))
            self.bestParams = np.array(list(dict(config_values).values())).astype(np.float64)
            config_file.close() # close config file once parameters are extracted

        # open the round file by creating a string with the round number if it does not already exist:
        round_filename = "round_" + str(self.round) + ".csv"
        self.round_path = str("C:\\Users\\oscar\\OneDrive\\Documents\\UNI_work\\Year_3\\Group_Project\\PSO_Training\\" + round_filename)
        self.headers = ["a1", "a2", "b1", "b2", "g1", "g2", "g3", "g4", "gEnd", "intNear", "intFar", "intThresh", # current values
                      "a1*", "a2*", "b1*", "b2*", "g1*", "g2*", "g3*", "g4*", "gEnd*", "intNear*", "intFar*", "intThresh*", # best values
                      "a1v", "a2v", "b1v", "b2v", "g1v", "g2v", "g3v", "g4v", "gEndv", "intNearv", "intFarv", "intThreshv", # inertial velocities
                      "currentFitness", "bestFitness"]

        if (self.round == 0 and self. index == 0):
        # If round 0, index 0, need to generate the initial params
            with open(self.round_path, 'w') as round_file:
                self.generateParamters(csv.DictWriter(round_file, self.headers))
                self.getStartPoint()
                config = pd.read_csv(self.conf_path)
                config.loc[int(np.where(config["Parameter"]=="X_Start")[0]),'Value'] = self.xStart
                config.loc[int(np.where(config["Parameter"]=="Y_Start")[0]),'Value'] = self.yStart
                config.loc[int(np.where(config["Parameter"]=="X_Target")[0]),'Value'] = self.xTarget
                config.loc[int(np.where(config["Parameter"]=="Y_Target")[0]),'Value'] = self.yTarget
                config.to_csv(self.conf_path, index=False)
                round_file.close()
        
        with open(self.round_path, 'r') as round_file:
            round_reader = csv.DictReader(round_file)
            temp = list(dict(list(round_reader)[self.index]).values())[:12]
            self.currentParams = [int(float(i)) for i in temp[0:2]] + [float(i) for i in temp[2:4]] + [int(float(i)) for i in temp[4:9]] + [float(i) for i in temp[9:11]] + [int(float(i)) for i in temp[11:12]]
            round_file.close()
            

    def __str__(self):
        # Controls what is returned by the object
        return f"TrainingSet({self.round},{self.index})"

    def generateParamters(self, writer):
        # Called when round 0, index 0 and the first set of parameters must be generated and saved to the CSV
        writer.writeheader()
        for i in range(self.population): # repeat for each particle in the population
            # Alpha, Beta, Gamma must be sorted on generation and relation must be maintained that e.g. g1<g2<g3<g4<gEnd
            a = sorted([randint(1,150), randint(1,150)])
            b = np.array(sorted([uniform(0.1,200.0),uniform(0.1,200.0)])).astype(np.float32) # ESP32 float is float32
            g = sorted([randint(1,100), randint(1,100), randint(1,100), randint(1,100), randint(1,100)])
            # For the integral gain, ensuring randomised mantissa and exponent
            i = np.array(np.array([round(random(),5), round(random(),5)])* 10.0**np.array([randint(-6,0), randint(-6,0)])).astype(np.float32)
            thresh = randint(0,150)
            params = {"a1": a[0], "a2": a[1], "b1": b[0], "b2": b[1], "g1": g[0], "g2": g[1], "g3": g[2], "g4":g[3] , "gEnd": g[4], "intNear": i[0], "intFar": i[1], "intThresh": thresh, # current values
                      "a1*": 0, "a2*": 0, "b1*": 0, "b2*": 0, "g1*": 0, "g2*": 0, "g3*": 0, "g4*": 0, "gEnd*": 0, "intNear*": 0, "intFar*": 0, "intThresh*": 0, # best values
                      "a1v": 0, "a2v": 0, "b1v": 0, "b2v": 0, "g1v": 0, "g2v": 0, "g3v": 0, "g4v": 0, "gEndv": 0, "intNearv": 0, "intFarv": 0, "intThreshv": 0, # inertial velocities
                      "currentFitness": 0, "bestFitness": float('inf')} # fitness behaviour
            writer.writerow(params)
        print("Round = 0, Index = 0. NEW PARAMETERS GENERATED")
        print("File: {}".format(self.round_path))
        return
    
    def getStartPoint(self):
        # Updates the start coordinate parameters with random params
        self.xStart = randint(10,245) # leave 10mm spacing around edge for camera cropping
        self.yStart = randint(10,245)
        self.xTarget = randint(10,245) 
        self.yTarget = randint(10,245)

    
    def uploadResult(self, result):
        # Accepts input from the other script giving outcome of the fitness function for the current set of parameters
        file = pd.read_csv(self.round_path)
        config = pd.read_csv(self.conf_path)
        file.loc[self.index, 'currentFitness'] = result # add the result of the fitness function to the file
        if (file.loc[self.index, 'currentFitness'] < file.loc[self.index, 'bestFitness']):
            # The particle has done better than it did previously, therefore need to update best values:
            file.loc[self.index, self.headers[12:24]] = np.array(file.loc[self.index, self.headers[:12]])
            file.loc[self.index, 'bestFitness'] = file.loc[self.index, 'currentFitness']
        file.to_csv(self.round_path, index=False) # overwrite the CSV with the updated data

        self.index = self.index + 1 # Increment the index
        if self.index == self.population:
            self.index = 0 # reset the index for the next round
            print("Round {} Complete, Please wait whilst the swarm moves...".format(self.round))
            current_round = pd.read_csv(self.round_path)
            temp_bestParams = self.bestParams
            for i in range(self.population):
                print("Particle {} is in motion".format(i))
                current = np.array(current_round.loc[i, self.headers[:12]])
                best = np.array(current_round.loc[i, self.headers[12:24]])
                velocities = np.array(current_round.loc[i, self.headers[24:36]])
                if (current_round.loc[i, 'currentFitness'] < self.bestCase):   
                    # Check for new global best             
                    print("\t * This particle currently minimises the objective function *")
                    self.bestCase = current_round.loc[i, 'currentFitness']
                    temp_bestParams = current
                # Randomness required for velocity update
                r1 = random()
                r2 = random()
                # Update the particle swarm on the basis of the velocity equation
                new_velocities = (self.W * velocities) + (self.c1 * r1 * (best - current)) + (self.c2 * r2 * (self.bestParams - current))
                # Need to make sure we maintain relationships such as a1 < a2, and ensure values aren't driven negative or out of range and 
                current_positions = current_round.loc[i, self.headers[:12]]
                new_positions = current_positions + new_velocities
                # Check alpha 1
                if (new_positions[0] <= 0): # check bottom end
                    new_positions[0] = 1
                elif (new_positions[0] > 150): # check upper limit
                    new_positions[0] = 150
                # Check alpha 2:
                if (new_positions[1] < new_positions[0]): # force alpha 1 < alpha 2
                    new_positions[1] = new_positions[0] + 1
                elif (new_positions[1] > 150):
                    new_positions[1] = 150
                # Check beta 1
                if (new_positions[2] <= 0): # beta only needs to check bottom end
                    new_positions[2] = 0.1
                # Check beta 2
                if (new_positions[3] < new_positions[2]): # beta 1 < beta 2
                    new_positions[3] = new_positions[2] + 0.1
                # Check gamma 1
                if (new_positions[4] <= 0):
                    new_positions[4] = 1
                elif (new_positions[4] >= 96):
                    new_positions[4] = 96 # this will make gamma end 100 and top of range
                # Check gamma 2
                if (new_positions[5] < new_positions[4]):
                    new_positions[5] = new_positions[4] + 1
                elif(new_positions[5] >= 97):
                    new_positions[5] = 97
                # Check gamma 3
                if (new_positions[6] < new_positions[5]):
                    new_positions[6] = new_positions[5] + 1
                elif(new_positions[6] >= 98):
                    new_positions[6] = 98
                # Check gamma 4
                if (new_positions[7] < new_positions[6]):
                    new_positions[7] = new_positions[6] + 1
                elif(new_positions[7] >= 99):
                    new_positions[7] = 99
                # Check gamma end
                if (new_positions[8] < new_positions[7]):
                    new_positions[8] = new_positions[7] + 1
                elif(new_positions[8] >= 100):
                    new_positions[8] = 100
                # Check int near
                if (new_positions[9] < 0.0000001):
                    new_positions[9] = 0.0000001
                # Check int far
                if (new_positions[10] < 0.0000001):
                    new_positions[10] = 0.0000001
                # Check int thresh
                if (new_positions[11] < 0):
                    new_positions[11] = 1
                elif (new_positions[11] > 150):
                    new_positions[11] = 150
                # Confirm the positions in the data frame
                actual_velocities = np.array(new_positions) - np.array(current_positions) # ensure the velocities don't think they are able to move out of the space
                current_round.loc[i, self.headers[24:36]] = actual_velocities # velocities must update
                current_round.loc[i, self.headers[:12]] = new_positions # positions must update

            # New round file creation
            self.bestParams = temp_bestParams # update the best params only after all particles have updated
            self.round = self.round + 1
            new_round = str("C:\\Users\\oscar\\OneDrive\\Documents\\UNI_work\\Year_3\\Group_Project\\PSO_Training\\" + "round_" + str(self.round) + ".csv")
            current_round.to_csv(new_round, index=False)
            # Update best case:
            config.loc[int(np.where(config["Parameter"]=="Global_Best")[0]),'Value'] = self.bestCase
            best_indexes = np.arange(int(np.where(config["Parameter"]=="Best_a1")[0]),int(np.where(config["Parameter"]=="Best_intThresh")[0])+1,1)
            config.loc[best_indexes,'Value'] = self.bestParams
            # Update the starting coordinates
            self.getStartPoint() # generate new starting coordinates for the next round.
            config.loc[int(np.where(config["Parameter"]=="X_Start")[0]),'Value'] = self.xStart
            config.loc[int(np.where(config["Parameter"]=="Y_Start")[0]),'Value'] = self.yStart
            config.loc[int(np.where(config["Parameter"]=="X_Target")[0]),'Value'] = self.xTarget
            config.loc[int(np.where(config["Parameter"]=="Y_Target")[0]),'Value'] = self.yTarget

        # Updating the config file with the round number and the index
        config.loc[int(np.where(config["Parameter"]=="Round")[0]),'Value'] = self.round
        config.loc[int(np.where(config["Parameter"]=="Index")[0]),'Value'] = self.index
        config.to_csv(self.conf_path, index=False)
        return 0

    def plotter(self):
        # Responsible for plotting some output data
        return
   

# for i in range(100):
#     training = PsoTraining()
#     print("Running: {}".format(training)) # Using the __str__ method in order to return useful data
#     params = training.currentParams
#     # print("Parameters: {}".format(params))
#     rand = uniform(0.0,1000.0)
#     training.uploadResult(rand)
