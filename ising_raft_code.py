"""
this code is to calculate the coverage of double layer ising model
"""
#！user/local/bin/python3.6
import math
import numpy as np
from numpy.random import random as rng
import random
import sys
EMPTY = 0
OCCUPIED = 1

def main():

    if len(sys.argv) < 9:
        print("ARGUMENT ERROR")
        quit()
    miu = float(sys.argv[1])  # chemical potential of the guests
    epsilon_rg = float(sys.argv[2])  # interaction between a receptor and guest bond
    membrane_size = int(sys.argv[3])  # membrane size (assume the membrane is square)
    guest_size_x = int(sys.argv[4])  # guest size on x axis
    guest_size_y = int(sys.argv[5])  # guest size on y axis
    epsilon_gg = float(sys.argv[6])  # interaction between a guest and a guest bond
    epsilon_rr = float(sys.argv[7])  # interaction between receptor and another receptor
    MINLRD = round(float(sys.argv[8]),2)

    print("miu is: {:.2f}".format(miu))
    print("epsilon_rg is: {:.2f}".format(epsilon_rg))
    print("epsilon_gg is: {:.2f}".format(epsilon_gg))
    print("epsilon_rr is: {:.2f}".format(epsilon_rr))
    print("membrane size is: {:d}".format(membrane_size))
    print("guest size x is: {:d}".format(guest_size_x))
    print("guest size y is: {:d}".format(guest_size_y))
    print("receptor density is: {:.2f}".format(MINLRD))

    step =10000
    equilibration =10000

    attached_guest = 1

    printoutstep =100
    vmdoutstep=100

    random.seed()

    file_name_result = "miu_" + str(miu) +"_mem_"+str(membrane_size)+ "_guest_" + str(guest_size_x) + "_" + str(guest_size_y) + "_epsilonrg_" \
                       + str(epsilon_rg) + "_epsilonrr_" + str(epsilon_rr) + "_epsilongg_" + str(epsilon_gg) +"lrd"+str(MINLRD)+".dat"
    file_name_vmd = "vmd_miu_" + str(miu) +"_mem_ojoisg"+str(membrane_size)+ "_guest_" + str(guest_size_x) + "_" + str(guest_size_y) + "_epsilon_rg" \
                    + str(epsilon_rg) + "_epsilonrr_" + str(epsilon_rr) + "_epsilongg_" + str(epsilon_gg) +"lrd"+str(MINLRD)+".xyz"

    for exponent in np.arange(MINLRD, MINLRD+1):
        receptor_guest_model = AdsorptionMonteCarlo()
        receptor_density = math.pow(10,exponent)
        print("density is: ", receptor_density)
        number_of_receptor = int(receptor_density * membrane_size ** 2)
        print("number of receptor is: {:d} \n".format(number_of_receptor))
        receptor_guest_model.set_parameters(epsilon_rg, epsilon_gg, epsilon_rr, membrane_size, guest_size_x,
                                            guest_size_y, attached_guest, number_of_receptor)

        initial = receptor_guest_model.get_list_guest_position()
        list_guest = initial[0]
        print("list_gust is: ", list_guest)
        dict_guest = initial[1]
        print("dict_guest: ", dict_guest)
        receptors_initial_config = receptor_guest_model.receptors_configuration()
        receptors = receptors_initial_config[0]
        receptor_list = receptors_initial_config[1]
        print("the initial configuration of receptors is: \n", receptors)

        initiate = receptor_guest_model.get_initial_guest_occupy(list_guest, dict_guest)
        guest_occupy = initiate[0]
        print("the initial guest lattice configuration is at\n", guest_occupy)
        guest_anchor_position = initiate[1]
        print("the position of the guest is: \n", guest_anchor_position)
        print("after initialization, the dict_guest is: ", dict_guest)
        # receptor_guest_model.number_of_bound_guest = receptor_guest_model.get_bound_guest(
        #     receptors,guest_occupy, guest_anchor_position)
        # print("the number of bound guest is: ", receptor_guest_model.number_of_bound_guest)
        receptor_guest_model(receptors, guest_occupy, guest_anchor_position, miu, file_name_result,
                             file_name_vmd, exponent,receptor_list, list_guest, dict_guest,step=step,
                             equilibration=equilibration, kbt=1.0, printoutstep=printoutstep,vmdoutstep=vmdoutstep)


class MonteCarlo(object):
    def __init__(self):
        self.kbt = 0.0

    def metropolis(self, delta_energy):
        return math.exp(-delta_energy/self.kbt) > rng()

    def set_temperature(self, kbt):
        self.kbt = kbt


class AdsorptionMonteCarlo(MonteCarlo):
    def __init__(self):
        super().__init__()
        self.epsilon_receptors_guest = 0.0
        self.epsilon_guest_guest = 0.0
        self.epsilon_receptor_receptor= 0.0
        self.membrane_size = 0
        self.number_of_attached_guest = 0
        self.number_of_bound_guest=0
        self.guest_size_x = 0
        self.guest_size_y = 0
        self.energies = None
        self.total_steps = 0
        self.sum_of_energies = 0.0
        self.sum_of_square_energies = 0.0
        self.sum_of_attached_guest = 0.0
        self.sum_of_bound_guest = 0.0
        self.sum_of_square_attached_guest = 0.0
        self.sum_of_square_bound_guest = 0.0
        self.number_of_receptors = 0

    def set_parameters(self, epsilon_receptors_guest, epsilon_guest_guest, epsilon_receptor_receptor,  membrane_size,
                       guest_size_x, guest_size_y, number_of_attached_guest, number_of_receptors):
        self.epsilon_receptors_guest = epsilon_receptors_guest
        self.epsilon_guest_guest = epsilon_guest_guest
        self.epsilon_receptor_receptor = epsilon_receptor_receptor
        self.membrane_size = membrane_size
        self.number_of_attached_guest = number_of_attached_guest
        self.number_of_receptors = number_of_receptors
        self.number_of_bound_guest = 0
        self.guest_size_x = guest_size_x
        self.guest_size_y = guest_size_y
        self.energies = None
        self.total_steps = 0
        self.sum_of_energies = 0.0
        self.sum_of_square_energies = 0.0
        self.sum_of_attached_guest = 0.0
        self.sum_of_square_attached_guest = 0.0
        self.sum_of_bound_guest = 0.0
        self.sum_of_square_bound_guest = 0.0

    def get_list_guest_position(self):
        list_guest = [(x * self.membrane_size + y)for x in range(self.membrane_size) if x % self.guest_size_y ==0
                               for y in range(self.membrane_size) if y % self.guest_size_x ==0]
        dict_guest = dict(zip(list_guest, len(list_guest)*[0]))
        return list_guest, dict_guest

    #  set the random initial configuration of the receptors on the membrane
    def receptors_configuration(self):
        receptors = np.zeros(self.membrane_size*self.membrane_size)  # first no receptor on the membrane
        a = np.arange(self.membrane_size*self.membrane_size)
        np.random.shuffle(a)
        receptor_list = list(a[:self.number_of_receptors])
        receptors[receptor_list] = 1
        return np.array(receptors), receptor_list

    def get_initial_guest_occupy(self, list_guest, dict_guest):
        # randomly choose a site from the list_guest
        guest_position = random.choice(list_guest)
        guest_occupy = np.zeros(self.membrane_size * self.membrane_size)
        guest_position_x = int(guest_position / self.membrane_size)
        guest_position_y = int(guest_position % self.membrane_size)
        current_guest_occupy = [(x % self.membrane_size * self.membrane_size + y % self.membrane_size)
                                for x in range(guest_position_x, guest_position_x + self.guest_size_x) \
                                for y in range(guest_position_y, guest_position_y + self.guest_size_y)]
        for number in current_guest_occupy:
            guest_occupy[number] = 1.0
        dict_guest[guest_position] = 1
        guest_anchor_position = [guest_position]
        return guest_occupy, guest_anchor_position
        #guest_occupy is a numpy array with 0 or 1 as elements, but guest anchor position is a list to sttore
        # the position of the guests

    def get_bound_guest(self, receptors,  guest_anchor_position):
        list_guest_receptor = []
        # guest_receptor = 0
        for anchor_position in guest_anchor_position:
            position_with_receptor = 0
            guest_position_x = int(anchor_position / self.membrane_size)
            guest_position_y = int(anchor_position % self.membrane_size)
            current_guest_occupy = [(x % self.membrane_size * self.membrane_size + y % self.membrane_size)
                                for x in range(guest_position_x, guest_position_x + self.guest_size_x) \
                                for y in range(guest_position_y, guest_position_y + self.guest_size_y)]
            for position in current_guest_occupy:
                position_with_receptor += receptors[position]
            if position_with_receptor != 0:
                list_guest_receptor.append(anchor_position)
        return list_guest_receptor

    def guest_guest_energy(self, guest_occupy,  guest_anchor_number):
        guest_energy = 0
        guest_position_x = int(guest_anchor_number / self.membrane_size)
        guest_position_y = int(guest_anchor_number % self.membrane_size)
        neighbour_up = [(guest_position_x-1)% self.membrane_size*self.membrane_size +y%self.membrane_size for
                     y in range(guest_position_y + self.guest_size_x)]
        neighbour_down = [(guest_position_x+self.guest_size_y)%self.membrane_size*self.membrane_size +
                          y %self.membrane_size for y in range(guest_position_y + self.guest_size_x)]
        neighbour_left = [x%self.membrane_size*self.membrane_size + (guest_position_y-1) %self.membrane_size for
                          x in range(guest_position_x, guest_position_x+self.guest_size_y)]
        neighbour_right = [x%self.membrane_size*self.membrane_size + (guest_position_y + self.guest_size_x)
                           %self.membrane_size for x in range(guest_position_x, guest_position_x+self.guest_size_y)]
        guest_neighbour = neighbour_up + neighbour_down + neighbour_left + neighbour_right
        for guest_number in guest_neighbour:
            if guest_occupy[guest_number]:
                guest_energy += 1
        guest_energy *= self.epsilon_guest_guest
        return guest_energy

    def total_guest_energy(self, guest_occupy, guest_anchor_position):
        sum_guest_energy = 0.0
        for guest_anchor_number in guest_anchor_position:
            sum_guest_energy += self.guest_guest_energy(guest_occupy, guest_anchor_number)
        return sum_guest_energy / 2.0

    def receptor_receptor_energy(self, receptors, receptor_number):
        if receptors[receptor_number] == 0.0:  # if the point is not occupied by the receptor, the energy is 0
            energy_receptor_receptor = 0.0
        else:
            receptor_position_x = int(receptor_number / self.membrane_size)
            receptor_position_y = int(receptor_number % self.membrane_size)
            left = [receptor_position_x, receptor_position_y-1]
            right = [receptor_position_x, receptor_position_y+1]
            up = [receptor_position_x-1, receptor_position_y]
            down = [receptor_position_x+1, receptor_position_y]
            lef_number = left[0]%self.membrane_size*self.membrane_size + left[1]%self.membrane_size
            right_number = right[0]%self.membrane_size*self.membrane_size + right[1]%self.membrane_size
            up_number = up[0] % self.membrane_size*self.membrane_size + up[1]%self.membrane_size
            down_number = down[0] % self.membrane_size * self.membrane_size + down[1] % self.membrane_size
            energy_receptor_receptor = (receptors[lef_number] + receptors[right_number] + receptors[down_number] + \
                                       receptors[up_number]) * self.epsilon_receptor_receptor
        return energy_receptor_receptor

    def total_receptor_energy(self, receptors):
        sum_receptor_energy = 0.0
        for receptor_number in range(receptors.size):
            sum_receptor_energy += self.receptor_receptor_energy(receptors, receptor_number)
        return sum_receptor_energy / 2.0

    def receptor_guest_energy_onepair_for_guest(self, receptors, guest_position_list):
        energy = 0
        for position in guest_position_list:
            energy += receptors[position]
        energy *= self.epsilon_receptors_guest
        return energy

    def total_get_receptor_guest_energy(self, guest_occupy, receptors):
        energy = np.sum(guest_occupy * receptors )
        # print("current energy is: \n", self.current_energy)
        return energy*self.epsilon_receptors_guest

    def receptor_guest_pair_energy_for_receptor(self, guest_occupy, receptors, receptor_position):
        return self.epsilon_receptors_guest * receptors[receptor_position] * guest_occupy[receptor_position]

    #  to insert an guest, first check if there are enough guest left in the solution and if the
    #  position randomly chosen is not occupied
    #  guest_occupied is an array to store the points covered by the guest
    #  guest_anchor_position is an array to store the left_up coordinate of the guest

    def insert_guest(self, receptors, guest_anchor_position, guest_occupy, miu, list_guest, dict_guest):
        # print("try to insert a guest")
        if len(guest_anchor_position )* self.guest_size_x * self.guest_size_y \
                < self.membrane_size * self.membrane_size:
            # check if there is enough space for another guest
            # guest_position = np.random.randint(receptors.size) #randomly choose a position to insesrt the guest
            guest_position = random.choice(list_guest) # choose a site randomly from list_guest
            guest_position_x = int(guest_position / self.membrane_size)
            guest_position_y = int(guest_position % self.membrane_size)
            insert_guest_occupy = [(x % self.membrane_size * self.membrane_size + y % self.membrane_size)
                                            for x in range(guest_position_x, guest_position_x + self.guest_size_y)
                                          for y in range(guest_position_y, guest_position_y + self.guest_size_x)]
            # check if the position is empty
            # if the site are not occupied
                # flip the position to 1
            if dict_guest[guest_position] == 0: # position empty
                for position in insert_guest_occupy:
                    guest_occupy[position] = 1.0
                delta_energy_rg = self.receptor_guest_energy_onepair_for_guest(receptors, insert_guest_occupy)
                delta_energy_gg = self.guest_guest_energy(guest_occupy, guest_position)
                delta_energy = delta_energy_rg+delta_energy_gg
                metropolis= (self.membrane_size ** 2 / (len(guest_anchor_position) + 1)* math.exp((-delta_energy + miu) / self.kbt))
                if metropolis > rng():
                    # add one element to the list of guest
                    # and the guest site is occupied
                    guest_anchor_position.append(guest_position)
                    dict_guest[guest_position] =1
                    # print("***INSERT****")
                    return True, delta_energy
                else:
                    # flip back the positions to 0
                    for position in insert_guest_occupy:
                        guest_occupy[position] = 0.0
                    return False, 0.0
            else:
                return False, 0.0
        else:
            return False, 0.0

    #     to remove a guest, choose from the guests on the membrane randomly and remove it
    #     according to the rule
    def remove_guest(self, receptors, guest_anchor_position, guest_occupy, miu, dict_guest):
        # check if there is guest on the membrane
        if len(guest_anchor_position) >= 1:
            remove_guest_index = np.random.randint(len(guest_anchor_position))
            remove_guest = guest_anchor_position[remove_guest_index]
            guest_anchor_position_x = int(remove_guest / self.membrane_size)
            guest_anchor_position_y = int(remove_guest % self.membrane_size)
            remove_guest_occupy = [(x % self.membrane_size * self.membrane_size + y % self.membrane_size)
                                   for x in range(guest_anchor_position_x, guest_anchor_position_x + self.guest_size_y)
                                   for y in range(guest_anchor_position_y, guest_anchor_position_y + self.guest_size_x)]
            # delta_energy_rg = - np.sum(receptors[remove_guest_occupy]) * self.epsilon_receptors_guest
            delta_energy_rg = -self.receptor_guest_energy_onepair_for_guest(receptors, remove_guest_occupy)
            delta_energy_gg = -self.guest_guest_energy(guest_occupy, remove_guest)
            delta_energy = delta_energy_rg + delta_energy_gg
            if (len(guest_anchor_position)/ self.membrane_size**2 *
                math.exp((-delta_energy - miu) / self.kbt)) > rng():
                # flip the positions to 0
                for position in remove_guest_occupy:
                    guest_occupy[position] = 0
                # delete the guest in the guest list and the site is empty
                del guest_anchor_position[remove_guest_index]
                dict_guest[remove_guest] = 0
                # print("&&&&&&&&&REMOVE&&&&&&&&&")
                return True, delta_energy
            else:
                return False, 0.0
        else:
            return False, 0.0

    def attempt_move_guest(self, receptors, guest_anchor_position, guest_occupy, list_guest, dict_guest):
        # print("try to move a guest")
        move_guest_index = np.random.randint(len(guest_anchor_position))
        move_guest=guest_anchor_position[move_guest_index]
        move_guest_x = int(move_guest / self.membrane_size)
        move_guest_y = int(move_guest % self.membrane_size)
        move_guest_occupy = [(x % self.membrane_size * self.membrane_size + y % self.membrane_size)
                                            for x in range(move_guest_x, move_guest_x + self.guest_size_y)
                                            for y in range(move_guest_y, move_guest_y + self.guest_size_x)]

        proposed_position_guest = random.choice(list_guest)
        guest_position_x = int(proposed_position_guest / self.membrane_size)
        guest_position_y = int(proposed_position_guest % self.membrane_size)
        proposed_position_guest_occupy = [(x % self.membrane_size * self.membrane_size + y % self.membrane_size)
                                            for x in range(guest_position_x, guest_position_x + self.guest_size_y)
                                            for y in range(guest_position_y, guest_position_y + self.guest_size_x)]
        # check if the position is occupied by guest

        # if the positions are not occupied, we calculate the energies related to the guest and then then put the guests
        # there and calculate the energy difference
        if dict_guest[proposed_position_guest] == 0:
            energy_rg_before = self.receptor_guest_energy_onepair_for_guest(receptors,move_guest_occupy)
            energy_gg_before = self.guest_guest_energy(guest_occupy, move_guest)
            # try move the guest which means to make the previous positions 0 and make the proposed position 1
            for position in move_guest_occupy:
                guest_occupy[position] = 0
            for position in proposed_position_guest_occupy:
                guest_occupy[position] = 1
            energy_rg_after = self.receptor_guest_energy_onepair_for_guest(receptors,proposed_position_guest_occupy)
            energy_gg_after = self.guest_guest_energy(guest_occupy, proposed_position_guest)
            delta_energy = energy_rg_after - energy_rg_before + energy_gg_after - energy_gg_before
            if self.metropolis(delta_energy):
                guest_anchor_position[move_guest_index] = proposed_position_guest
                dict_guest[proposed_position_guest] = 1
                dict_guest[move_guest] = 0
                # print("%%%%%MOVE%%%%%")
                return True, delta_energy
            else:
                # move the guest back to original position
                for position in move_guest_occupy:
                    guest_occupy[position] = 1
                for position in proposed_position_guest_occupy:
                    guest_occupy[position] = 0
                return False, 0.0
        else:
            return False, 0.0

    def attempt_move_receptors(self, receptors, guest_occupy, receptor_list):
    #randomly choose a receptor, which is the position of the receptor
        move_index = np.random.randint(len(receptor_list))
        receptor_old_position= receptor_list[move_index]
        # randomly choose a position to put the receptors
        receptor_new_position = np.random.randint(receptors.size)
        if receptors[receptor_new_position] == 1.0:
            return False, 0.0, 0.0, 0.0, receptors, 0.0
        else:
            energy_pre_rg = self.receptor_guest_pair_energy_for_receptor(guest_occupy, receptors, receptor_old_position)
            energy_pre_rr = self.receptor_receptor_energy(receptors, receptor_old_position)
        # swap the two receptors
            receptors[receptor_old_position] = 0.0
            receptors[receptor_new_position] = 1.0
            energy_after_rg = self.receptor_guest_pair_energy_for_receptor(guest_occupy,receptors,receptor_new_position)
            energy_after_rr = self.receptor_receptor_energy(receptors, receptor_new_position)
            delta_energy_rr = energy_after_rr - energy_pre_rr
            delta_energy_rg = energy_after_rg - energy_pre_rg
            delta_energy = delta_energy_rr + delta_energy_rg
            if self.metropolis(delta_energy):
                receptor_list[move_index] = receptor_new_position
                return True, receptor_old_position, receptor_old_position,receptors,delta_energy
            else:
            # swap back two receptors
                receptors[receptor_old_position] = 1.0
                receptors[receptor_new_position] = 0.0
                return False, 0.0, 0.0, 0.0, receptors, 0.0

    def update_observables(self, current_energy, guest_anchor_position):
        self.total_steps += 1
        self.sum_of_energies += current_energy
        self.sum_of_square_energies += current_energy**2
        current_number_of_attached_guest = len(guest_anchor_position) * self.guest_size_x* self.guest_size_y\
                                           / (self.membrane_size**2)
        self.sum_of_attached_guest += current_number_of_attached_guest
        self.sum_of_square_attached_guest += current_number_of_attached_guest**2

    def update_bound_guest(self):
        self.sum_of_bound_guest += self.number_of_bound_guest
        self.sum_of_square_bound_guest += self.number_of_bound_guest**2

    def get_observables(self):
        average_energy = self.sum_of_energies / self.total_steps
        average_square_energy = self.sum_of_square_energies / self.total_steps
        average_square_energy_error = math.sqrt((average_square_energy - average_energy * average_energy) \
                / self.total_steps)
        average_of_attached_guest = self.sum_of_attached_guest / self.total_steps
        average_of_square_attached_guest = self.sum_of_square_attached_guest / self.total_steps
        average_of_square_attached_guest_error = \
            math.sqrt(abs(average_of_square_attached_guest - average_of_attached_guest**2) / self.total_steps)
        average_of_bound_guest = self.sum_of_bound_guest / self.total_steps
        average_of_square_bound_guest = self.sum_of_square_bound_guest / self.total_steps
        average_of_square_bound_guest_error = \
            math.sqrt(abs(average_of_square_bound_guest - average_of_bound_guest*average_of_bound_guest) \
                      / self.total_steps)
        return average_energy, average_square_energy_error,\
               average_of_attached_guest, average_of_square_attached_guest_error,\
               average_of_bound_guest, average_of_square_bound_guest_error

    def reset_observables(self):
        self.total_steps = 0
        self.sum_of_energies = 0.0
        self.sum_of_square_energies = 0.0
        self.sum_of_attached_guest = 0.0
        self.sum_of_square_attached_guest = 0.0
        self.sum_of_bound_guest = 0.0
        self.sum_of_square_bound_guest = 0.0

    @staticmethod
    def write_to_file(file_name):
        f = open(file_name, "a")
        return f

    def __call__(self, receptors, guest_occupy, guest_anchor_position, miu, file_name_guests, file_name_trajectory,
                 receptor_density, receptor_list, list_guest, dict_guest,
                 step=1, equilibration=1, kbt=1.0, printoutstep=100,vmdoutstep=1000):
        self.reset_observables()
        self.set_temperature(kbt)
        f1 = self.write_to_file(file_name=file_name_guests)
        f2 = self.write_to_file(file_name=file_name_trajectory)
        current_energy = self.total_get_receptor_guest_energy(guest_occupy, receptors) \
                         +self.total_receptor_energy(receptors)+\
                         self.total_guest_energy(guest_occupy, guest_anchor_position)
        print("initial energy is: {:10.4f} \n".format(current_energy))
        # exponent = math.log10(np.count_nonzero(receptors) / receptors.size)
        number_of_trail_move_receptor = 0
        number_of_trail_move_guest = 0
        number_of_trail_insert = 0
        number_of_trail_remove = 0

        number_of_accepted_move_receptor = 0
        number_of_accepted_move_guest = 0
        number_of_accepted_insert = 0
        number_of_accepted_remove = 0

        print("beginning equilibration...")
        print("start config", guest_occupy)
        for i in range(equilibration):
            # for move_receptor in range(receptors.size):
            for move_receptor in range(self.number_of_receptors):
                result = self.attempt_move_receptors(receptors, guest_occupy, receptor_list)
                # print("result", result)
                number_of_trail_move_receptor += 1
                if result[0]:
                    number_of_accepted_move_receptor += 1
                current_energy += result[-1]

            for move_guest in range(len(guest_anchor_position)):
                result = self.attempt_move_guest(receptors, guest_anchor_position,guest_occupy, list_guest, dict_guest)
                number_of_trail_move_guest +=1
                # print("move a guest", result)
                if result[0]:
                    number_of_accepted_move_guest += 1
                    # print("after move guest", guest_occupy)
                    # print("guest position", guest_anchor_position)
                current_energy += result[-1]

            if rng() > 0.5:
                number_of_trail_insert += 1
                insert = self.insert_guest(receptors, guest_anchor_position, guest_occupy, miu,list_guest, dict_guest)
                # print("insert", insert)
                if insert[0]:
                    number_of_accepted_insert += 1
                    # print("after insert guest", guest_occupy)
                    # print("guest position", guest_anchor_position)
                current_energy += insert[-1]

            else:
                number_of_trail_remove += 1
                remove = self.remove_guest(receptors, guest_anchor_position, guest_occupy, miu, dict_guest)
                # print("remove", remove)
                if remove[0]:
                    number_of_accepted_remove += 1
                    # print("after remove guest", guest_occupy)
                    # print("guest position", guest_anchor_position)
                current_energy += remove[-1]
            if i % printoutstep == 0:
                print("equilibration step {:d} ".format(i))
        if number_of_trail_move_receptor:
            ratio_move_receptors = number_of_accepted_move_receptor / number_of_trail_move_receptor
        else:
            ratio_move_receptors = 0.0
        if number_of_trail_move_guest:
            ratio_move_guests = number_of_accepted_move_guest / number_of_trail_move_guest
        else:
            ratio_move_guests = 0.0
        if number_of_trail_insert:
            ratio_insert_guests = number_of_accepted_insert / number_of_trail_insert
        else:
            ratio_insert_guests = 0.0
        if number_of_trail_remove:
            ratio_remove_guest = number_of_accepted_remove / number_of_trail_remove
        else:
            ratio_remove_guest = 0.0
        print("accepted move receptors is {:.10f} accepted move guests is {:.10f}"
              "accepted insert guests is {:.10f} accepted remove guests is {:.10f}".format(ratio_move_receptors,
                                                                                           ratio_move_guests,
                                                                                           ratio_insert_guests,
                                                                                           ratio_remove_guest))

        print("complete equilibration")
        number_of_trail_move_receptor = 0
        number_of_trail_move_guest = 0
        number_of_trail_insert = 0
        number_of_trail_remove = 0

        number_of_accepted_move_receptor = 0
        number_of_accepted_move_guest = 0
        number_of_accepted_insert = 0
        number_of_accepted_remove = 0
        self.reset_observables()
        print("Energy after equilibration is: ", current_energy)
        print("beginning calculation")

        for j in range(step):
            for move_receptor in range(self.number_of_receptors):
                number_of_trail_move_receptor += 1
                result = self.attempt_move_receptors(receptors, guest_occupy, receptor_list)
                if result[0]:
                    number_of_accepted_move_receptor += 1
                current_energy += result[-1]
            for move_guest in range(len(guest_anchor_position)):
                number_of_trail_move_guest += 1
                result = self.attempt_move_guest(receptors, guest_anchor_position, guest_occupy, list_guest, dict_guest)
                if result[0]:
                    number_of_accepted_move_guest += 1
                current_energy += result[-1]

            if rng() > 0.5:
                number_of_trail_insert += 1.0
                insert = self.insert_guest(receptors, guest_anchor_position, guest_occupy, miu, list_guest, dict_guest)
                if insert[0]:
                    number_of_accepted_insert += 1
                current_energy += insert[-1]

            else:
                number_of_trail_remove += 1
                remove = self.remove_guest(receptors, guest_anchor_position, guest_occupy, miu, dict_guest)
                if remove[0]:
                    number_of_accepted_remove += 1
                current_energy += remove[-1]

            # print("guest_occupy is:", guest_occupy)
            # print("number of guests is",len(guest_anchor_position))
            # print("guest anchor position is", guest_anchor_position)
            self.update_observables(current_energy, guest_anchor_position)  # update observables
            guest_bound_result = self.get_bound_guest(receptors, guest_anchor_position)
            self.number_of_bound_guest = len(guest_bound_result) * self.guest_size_x * self.guest_size_y \
                                         / (self.membrane_size ** 2)
            self.update_bound_guest()
            if j % printoutstep == 0:
                print("*******calculation steps {:d} energy {:.10f} +-{:.10f} number_of_attached_guest "
                      "{:.10f} +-{:.10f} number_of_bound_guests " "{:.10f}+-{:.10f}\n"
                      .format(j, *self.get_observables()))
            if j % vmdoutstep == 0:
                f2.write(str(2 * self.membrane_size*self.membrane_size) + "\n")
                f2.write("\n")
                # for i in range(receptors.size):
                for i in range(self.membrane_size * self.membrane_size):
                    x = int(i / self.membrane_size)
                    y = int(i % self.membrane_size)
                    if receptors[i]:
                        f2.write(str("O   ") + str(x) + "   " + str(y) + "   " + str("0    \n"))
                    else:
                        f2.write(str("O   ") + str(x) + "   " + str(y) + "   " + str("-1    \n"))
                for k in range(self.membrane_size * self.membrane_size):
                    x = int(k / self.membrane_size)
                    y = int(k % self.membrane_size)
                    if guest_occupy[k]:
                        f2.write(str("C   ") + str(x) + "   " + str(y) + "   " + str("2    \n"))
                    else:
                        f2.write(str("C   ") + str(x) + "   " + str(y) + "   " + str("-2    \n"))

                # this is to get the guests bound with receptors
                # one_dimension_guest_bound_size = len(guest_bound_result)*self.guest_size_y*self.guest_size_x
                # for k in range(receptors.size):
                #     x = int(k / self.membrane_size)
                #     y = k % self.membrane_size
                #     if k in range(one_dimension_guest_bound_size):
                #         f2.write(str("C   ") + str(x) + "   " + str(y) + "   " + str("2    \n"))
                #     else:
                #         f2.write(str("C   ") + str(x) + "   " + str(y) + "   " + str("-2    \n"))

        if number_of_trail_move_receptor:
            ratio_move_receptors = number_of_accepted_move_receptor / number_of_trail_move_receptor
        else:
            ratio_move_receptors = 0.0
        if number_of_trail_move_guest:
            ratio_move_guests = number_of_accepted_move_guest / (number_of_trail_move_guest)
        else:
            ratio_move_guests = 0.0
        if number_of_trail_insert:
            ratio_insert_guests = number_of_accepted_insert / number_of_trail_insert
        else:
            ratio_insert_guests = 0.0
        if number_of_trail_remove:
            ratio_remove_guest = number_of_accepted_remove / number_of_trail_remove
        else:
            ratio_remove_guest = 0.0
        print("accepted move receptors is {:.10f} accepted move guests is {:.10f}"
              "accepted insert guests is {:.10f} accepted remove guests is {:.10f}".format(ratio_move_receptors,
                                                                                           ratio_move_guests,
                                                                                           ratio_insert_guests,
                                                                                           ratio_remove_guest))

        result_get_observable = self.get_observables()
        f1.write(str(receptor_density) + " " + str(result_get_observable[0]) + " " +
                 str(result_get_observable[1]) + " " + str(math.log10(result_get_observable[2]) if \
                     result_get_observable[2] !=0 else -100) + " " + str(result_get_observable[3]) + " " +
                 str(math.log10(result_get_observable[4]) if result_get_observable[4] != 0 else -100) +
                 " " + str(result_get_observable[5])+"\n")

        print("Production complete")
        f1.close()
        f2.close()


if __name__ == "__main__":
    main()











