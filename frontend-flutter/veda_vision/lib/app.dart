import 'package:flutter/material.dart';
import 'theme/app_colors.dart';
import 'screens/home_page.dart';

class AyurvedaApp extends StatelessWidget {
  const AyurvedaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ayurveda Plant Recognition',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        fontFamily: 'Segoe UI',
        colorScheme: ColorScheme.fromSeed(seedColor: AppColors.titleColor),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}