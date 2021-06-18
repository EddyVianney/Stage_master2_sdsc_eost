#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:48:40 2021

@author: eost-user
"""
from TimeSerie import TimeSerie

class Pixel(TimeSerie):
    
    def __init__(self, id_, lat, lon, topo, ns, ew, ns_vel, ew_vel):
        super(TimeSerie, self).__init__()
        self.id = id_
        self.lat = lat
        self.lon = lon
        self.topo = topo
        self.ns = self.impute(ns)
        self.ew = self.impute(ew)
        self.ns_vel = ns_vel
        self.ew_vel = ew_vel
        self.is_selected = False
        
    def has_enough_values(self, pc):
        return self.compute_null_val_percentage(self.ns) < pc
    
    def is_ns_linear_regression_significant(self, alpha):
        return self.compute_linear_reg_pval(self.ns) < alpha
        
    def is_ew_linear_regression_significant(self, alpha):
        return self.compute_linear_reg_pval(self.ew) < alpha
    
    def is_linear_regression_significant(self, alpha):
        return self.is_ns_linear_regression_significant(alpha) or self.is_ew_linear_regression_significant(alpha)
    
    def is_steep(self, filename, ref, min_slope):
        return self.compute_slope(ref, filename) > min_slope
    
    def compute_vel(self, ns_component, ew_component):
        return math.sqrt(ns_component * ns_component + ew_component * ew_component)
    
    def is_moving(self, std, factor):
        return self.get_mean_velocity() > factor*std
    
    def is_to_select(self, filename, ref, alpha, min_slope, std, factor, pc):
        return self.is_linear_regression_significant(alpha) and self.is_moving(std, factor) and self.is_steep(filename, ref, min_slope)
    
    def compute_slope(self, ref, file):
        val = os.popen('gdallocationinfo -valonly -%s %s %f %f' % (ref, file, self.lat, self.lon)).read()
        if len(val) == 0:
            raise ValueError('La pente est non valide !')
        return  float(val)
    
    def get_mean_velocity(self):
        return math.sqrt(self.ns_vel*self.ns_vel + self.ew_vel*self.ew_vel)