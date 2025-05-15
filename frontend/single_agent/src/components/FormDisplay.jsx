import React, { useState } from 'react';

//当用户提交的是表单数据时，组件会显示
function FormComponent({ formData, onSubmit, messageId }) {
    const [formValues, setFormValues] = useState(formData.form_data || {});
    const [errors, setErrors] = useState({});
  
    const handleInputChange = (key, value) => {
      setFormValues((prev) => ({ ...prev, [key]: value }));
      if (errors[key]) {
        setErrors((prev) => ({ ...prev, [key]: '' }));
      }
    };
  
    const validateForm = () => {
      const newErrors = {};
      formData.form.required.forEach((field) => {
        if (!formValues[field] || formValues[field].trim() === '') {
          newErrors[field] = `${formData.form.properties[field].title} is required`;
        }
      });
      setErrors(newErrors);
      return Object.keys(newErrors).length === 0;
    };
  
    const handleSubmit = (e) => {
      e.preventDefault();
      if (validateForm()) {
        onSubmit(messageId, formValues);
      }
    };
  
    return (
      <div className="p-4 bg-gray-100 rounded-lg shadow-md">
        {formData.instructions && (
          <h3 className="text-lg font-semibold mb-4">{formData.instructions}</h3>
        )}
        <form onSubmit={handleSubmit} className="space-y-4">
          {Object.entries(formData.form.properties).map(([key, prop]) => (
            <div key={key} className="flex flex-col">
              <label className="text-sm font-medium text-gray-700 mb-1">
                {prop.title}
                {formData.form.required.includes(key) && (
                  <span className="text-red-500">*</span>
                )}
              </label>
              <input
                type={prop.format === 'date' ? 'date' : prop.format === 'number' ? 'number' : 'text'}
                value={formValues[key] || ''}
                onChange={(e) => handleInputChange(key, e.target.value)}
                className={`p-2 border rounded-md focus:ring-indigo-500 focus:border-indigo-500 ${
                  errors[key] ? 'border-red-500' : 'border-gray-300'
                }`}
                placeholder={prop.description}
                required={formData.form.required.includes(key)}
              />
              {errors[key] && (
                <p className="text-red-500 text-xs mt-1">{errors[key]}</p>
              )}
            </div>
          ))}
          <div className="flex justify-end space-x-2">
            <button
              type="submit"
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded-md"
            >
              Submit
            </button>
          </div>
        </form>
      </div>
    );
  }
  

export default FormComponent;