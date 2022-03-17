**BR2-simulator** is the python tool to simulate assembly of FREE arm. To learn more details on this soft-robotics project, please visit `Gazzola Lab`_ and `Monolithic Systems Lab`_.

The goal of this package is to provide simple pipeline to test the physical kinematics and configuration of the soft-arm, built based on FREE actuator. The tool is a wrapper around a `PyElastica <https://docs.cosseratrods.org/en/latest/>`_ project; all the simulation models here are based on Cosserat Rod Theory. This method allow us to develop fast and easy environment to aid modeling, manufacturing/fabrication, and control.

.. Gallary
   -------

Installation
------------

The easist way to install the package is with pip:

.. code:: console

    $ pip install br2    

or download the source code from the `GitHub repo`_. If you are using the source code, make sure you install all the required dependencies in `requirements.txt`.



.. _Github repo: https://github.com/skim0119/BR2-simulator
.. _Monolithic Systems Lab: https://monolithicsystemslab.ise.illinois.edu
.. _Gazzola Lab: https://mattia-lab.com/
