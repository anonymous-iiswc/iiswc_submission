LIBDIR := libpimsim
APPDIR := apps

.PHONY: debug perf dramsim3_integ clean $(LIBDIR) $(APPDIR)
.DEFAULT_GOAL := perf

debug: $(LIBDIR) $(APPDIR)
	@echo "\nINFO: Built PIM Functional Simulator with target = debug\n"

perf: $(LIBDIR) $(APPDIR)
	@echo "\nINFO: Built PIM Functional Simulator with target = perf\n"

dramsim3_integ: $(LIBDIR) $(APPDIR)
	@echo "\nINFO: Built PIM Functional Simulator with target = dramsim3_integ\n"

clean: $(LIBDIR) $(APPDIR)

# Run "make PIM_SIM_TARGET=<PimDeviceEnum>" to override default simulation target
PIM_SIM_TARGET ?= PIM_DEVICE_NONE

$(LIBDIR) $(APPDIR):
	$(MAKE) -C $@ $(MAKECMDGOALS) PIM_SIM_TARGET=$(PIM_SIM_TARGET)

