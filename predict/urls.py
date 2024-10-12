from django.urls import path
from .views import  predict_heart_failure, results_view

urlpatterns = [
    # path('', home_view, name='home'),  # Ensure this points to home_view
    path('', predict_heart_failure, name='home'),
    path('results/', results_view, name='results'),  # Results page
    # path('results/', results_view, name='results'),
]
