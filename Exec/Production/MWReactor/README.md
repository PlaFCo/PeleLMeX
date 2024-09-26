## Microwave reactor
Extended the "EnclosedInjection" test case to include:
1. A embedded cylinder boundary condition;
2. An outlet at zlo and a wall at zhi;
3. The union of the previous cylinder with another along x;
4. An inlet of O2 at that horizontal cylinder;
5. Two tangential square pipe inlets of N2 on the top of the main pipe.

Currently using the input.3d-regt_react_heating
It was using the airthermalchemistry branch of PelePhysics but I am not sure how
to use this in the git submodule way. So I changed it back to the simpler air
scheme, see GNUmakefile, which does not have reactions.

It is using the spark framework but the source term is constant.
spark1.power defines the power, radius defines the shape of the source
term which is always an ellipsoid. The power increases from zero at spark1.time
until the defined power at t=spark1.duration.

For low powers (50) things work OK while for higher powers 200 the timestep
continually decreases to the lower limite.

Currently, the source term is constant over the ellipsoid. Before it was 
parabolic meaning that it would fall with r^2 until the edge of the ellispoid. 
See previous commit for the parabolic source term.
