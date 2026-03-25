{% if fullname == 'zrad.filtering.filtering_definitions' %}
Filter Definitions
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
