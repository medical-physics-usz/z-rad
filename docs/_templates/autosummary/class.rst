{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% set public_methods = methods | reject('equalto', '__init__') | list %}
{% if public_methods %}
.. rubric:: Methods

.. autosummary::
{% for item in public_methods %}
   ~{{ name }}.{{ item }}
{% endfor %}

{% for item in public_methods %}
.. automethod:: {{ name }}.{{ item }}

{% endfor %}
{% endif %}
{% set properties = properties_by_class.get(fullname, []) %}
{% if properties %}
.. rubric:: Properties

{% for item in properties %}
.. autoattribute:: {{ name }}.{{ item }}

{% endfor %}
{% endif %}
