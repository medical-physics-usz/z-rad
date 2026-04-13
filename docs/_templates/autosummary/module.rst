{% if fullname == 'zrad.filtering.base' %}
Filtering Base
==============
{% elif fullname == 'zrad.filtering.factory' %}
Filter Factory
==============
{% elif fullname == 'zrad.filtering.spatial' %}
Spatial Filters
===============
{% elif fullname == 'zrad.filtering.wavelet' %}
Wavelet Filters
==================
{% elif fullname == 'zrad.radiomics.radiomics_definitions' %}
Radiomics Definitions
=====================
{% else %}
{{ fullname.split('.')[-1] | escape }}
{{ fullname.split('.')[-1] | escape | underline }}
{% endif %}

.. automodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree: .
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
