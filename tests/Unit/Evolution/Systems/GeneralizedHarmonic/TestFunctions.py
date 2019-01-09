# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Test functions for normal dot fluxes


def spacetime_metric_normal_dot_flux(spacetime_metric, pi, phi, gamma1, gamma2,
                                     lapse, shift, inverse_spatial_metric,
                                     unit_normal):
    return - (1. + gamma1) * np.dot(shift, unit_normal) * spacetime_metric


def pi_normal_dot_flux(spacetime_metric, pi, phi, gamma1, gamma2, lapse, shift,
                       inverse_spatial_metric, unit_normal):
    return - np.dot(shift, unit_normal) * pi  \
        + lapse * np.einsum("ki,k,iab->ab",
                            inverse_spatial_metric, unit_normal, phi) \
           - gamma1 * gamma2 * np.dot(shift, unit_normal) * spacetime_metric


def phi_dot_flux(spacetime_metric, pi, phi, gamma1, gamma2, lapse, shift,
                 inverse_spatial_metric, unit_normal):
    return - np.dot(shift, unit_normal) * phi \
        + lapse * np.einsum("i,ab->iab", unit_normal, pi) \
           - gamma2 * lapse * np.einsum("i,ab->iab", unit_normal,
                                        spacetime_metric)

# End test functions for normal dot fluxes

# Test functions for gauge constraint


def gauge_constraint(gauge_function, spacetime_normal_one_form,
                     spacetime_normal_vector, inverse_spatial_metric,
                     inverse_spacetime_metric, pi, phi):
    # Sums only go over spatial indices of second index of phi
    phi_ija = phi[:, 1:, :]
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    constraint = gauge_function \
        + np.einsum("ij,ija->a", inverse_spatial_metric, phi_ija) \
        + np.einsum("b,ba->a", spacetime_normal_vector, pi) \
        - 0.5 * np.insert(np.einsum("bc,ibc->i", inverse_spacetime_metric, phi),
                          0, np.zeros(phi[0][0][0].shape)) \
        - 0.5 * np.einsum("a,i,bc,ibc->a", spacetime_normal_one_form,
                          spacetime_normal_vector_I,
                          inverse_spacetime_metric, phi) \
        - 0.5 * np.einsum("a,bc,bc->a", spacetime_normal_one_form,
                          inverse_spacetime_metric, pi)
    return constraint

# End test functions for gauge constraint

# Test functions for two-index constraint


def two_index_constraint_term_1_of_11(inverse_spatial_metric, d_phi):
    d_phi_jika = d_phi[:, :, 1:, :]
    return np.einsum("jk,jika->ia", inverse_spatial_metric, d_phi_jika)


def two_index_constraint_term_2_of_11(spacetime_normal_vector,
                                      spacetime_normal_one_form,
                                      inverse_spacetime_metric, d_phi):
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    d_phi_aicd = np.pad(d_phi, ((1, 0), (0, 0), (0, 0), (0, 0)), 'constant')
    term = np.einsum("cd,aicd->ia", inverse_spacetime_metric, d_phi_aicd)
    term += np.einsum("j,a,cd,jicd->ia", spacetime_normal_vector_I,
                      spacetime_normal_one_form, inverse_spacetime_metric,
                      d_phi)
    return -0.5 * term


def two_index_constraint_term_3_of_11(spacetime_normal_vector, d_pi):
    return np.einsum("b,iba->ia", spacetime_normal_vector, d_pi)


def two_index_constraint_term_4_of_11(spacetime_normal_one_form,
                                      inverse_spacetime_metric, d_pi):
    return -0.5 * np.einsum("cd,icd,a->ia", inverse_spacetime_metric, d_pi,
                            spacetime_normal_one_form)


def two_index_constraint_term_5_of_11(d_gauge_function):
    return d_gauge_function


def two_index_constraint_term_6_of_11(spacetime_normal_vector,
                                      spacetime_normal_one_form, phi,
                                      inverse_spacetime_metric):
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    phi_acd = np.pad(phi, ((1, 0), (0, 0), (0, 0)), 'constant')
    term = np.einsum("acd,ief,ce,df->ia", phi_acd, phi,
                     inverse_spacetime_metric,
                     inverse_spacetime_metric)
    term += np.einsum("j,a,jcd,ief,ce,df->ia", spacetime_normal_vector_I,
                      spacetime_normal_one_form, phi, phi,
                      inverse_spacetime_metric,
                      inverse_spacetime_metric)
    return 0.5 * term


def two_index_constraint_term_7_of_11(inverse_spatial_metric, phi,
                                      inverse_spacetime_metric,
                                      spacetime_normal_vector,
                                      spacetime_normal_one_form):
    phi_ike = phi[:, 1:, :]
    return 0.5 * np.einsum("jk,jcd,ike,cd,e,a->ia", inverse_spatial_metric, phi,
                           phi_ike, inverse_spacetime_metric,
                           spacetime_normal_vector,
                           spacetime_normal_one_form)


def two_index_constraint_term_8_of_11(inverse_spatial_metric, phi):
    phi_jma = phi[:, 1:, :]
    phi_ikn = phi[:, 1:, 1:]
    return -1.0 * np.einsum("jk,mn,jma,ikn->ia", inverse_spatial_metric,
                            inverse_spatial_metric, phi_jma, phi_ikn)


def two_index_constraint_term_9_of_11(phi, pi, spacetime_normal_one_form,
                                      inverse_spacetime_metric,
                                      spacetime_normal_vector):
    term = 0.5 * np.einsum("icd,be,a,cb,de->ia", phi, pi,
                           spacetime_normal_one_form, inverse_spacetime_metric,
                           inverse_spacetime_metric)
    term += 0.25 * np.einsum("icd,be,a,be,c,d->ia", phi, pi,
                             spacetime_normal_one_form,
                             inverse_spacetime_metric,
                             spacetime_normal_vector, spacetime_normal_vector)
    return term


def two_index_constraint_term_10_of_11(phi, pi, spacetime_normal_vector,
                                       inverse_spacetime_metric):
    term = -1.0 * np.einsum("icd,ba,c,bd->ia", phi, pi,
                            spacetime_normal_vector, inverse_spacetime_metric)
    term -= 0.5 * np.einsum("icd,ba,c,b,d->ia", phi, pi,
                            spacetime_normal_vector, spacetime_normal_vector,
                            spacetime_normal_vector)
    return term


def two_index_constraint_term_11_of_11(gamma2, spacetime_normal_one_form,
                                       inverse_spacetime_metric,
                                       spacetime_normal_vector,
                                       three_index_constraint):
    term = 0.5 * gamma2 * np.einsum("a,cd,icd->ia",
                                    spacetime_normal_one_form,
                                    inverse_spacetime_metric,
                                    three_index_constraint)
    term -= gamma2 * np.einsum("d,iad->ia",
                               spacetime_normal_vector, three_index_constraint)
    return term


def two_index_constraint(d_gauge_function, spacetime_normal_one_form,
                         spacetime_normal_vector, inverse_spatial_metric,
                         inverse_spacetime_metric, pi, phi, d_pi, d_phi, gamma2,
                         three_index_constraint):
    constraint = two_index_constraint_term_1_of_11(
        inverse_spatial_metric, d_phi)
    constraint += two_index_constraint_term_2_of_11(spacetime_normal_vector,
                                                    spacetime_normal_one_form,
                                                    inverse_spacetime_metric,
                                                    d_phi)
    constraint += two_index_constraint_term_3_of_11(
        spacetime_normal_vector, d_pi)
    constraint += two_index_constraint_term_4_of_11(spacetime_normal_one_form,
                                                    inverse_spacetime_metric,
                                                    d_pi)
    constraint += two_index_constraint_term_5_of_11(d_gauge_function)
    constraint += two_index_constraint_term_6_of_11(spacetime_normal_vector,
                                                    spacetime_normal_one_form,
                                                    phi,
                                                    inverse_spacetime_metric)
    constraint += two_index_constraint_term_7_of_11(inverse_spatial_metric, phi,
                                                    inverse_spacetime_metric,
                                                    spacetime_normal_vector,
                                                    spacetime_normal_one_form)
    constraint += two_index_constraint_term_8_of_11(
        inverse_spatial_metric, phi)
    constraint += two_index_constraint_term_9_of_11(phi, pi,
                                                    spacetime_normal_one_form,
                                                    inverse_spacetime_metric,
                                                    spacetime_normal_vector)
    constraint += two_index_constraint_term_10_of_11(
        phi, pi, spacetime_normal_vector, inverse_spacetime_metric)
    constraint += two_index_constraint_term_11_of_11(gamma2,
                                                     spacetime_normal_one_form,
                                                     inverse_spacetime_metric,
                                                     spacetime_normal_vector,
                                                     three_index_constraint)
    return constraint

# End test functions for two-index constraint
