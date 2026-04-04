import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext(null);

const FIREBASE_API_KEY = process.env.REACT_APP_FIREBASE_API_KEY;
const FIREBASE_AUTH_BASE = 'https://identitytoolkit.googleapis.com/v1';

function firebaseAuthUrl(endpoint) {
  if (!FIREBASE_API_KEY) {
    throw new Error('Firebase API key is missing. Check frontend/.env.local');
  }
  return `${FIREBASE_AUTH_BASE}/${endpoint}?key=${FIREBASE_API_KEY}`;
}

function mapFirebaseError(error, fallbackMessage) {
  const code = error?.response?.data?.error?.message;
  const messages = {
    EMAIL_EXISTS: 'Email is already registered',
    INVALID_EMAIL: 'Invalid email address',
    INVALID_PASSWORD: 'Incorrect password',
    EMAIL_NOT_FOUND: 'No account found for this email',
    USER_DISABLED: 'This account is disabled',
    TOO_MANY_ATTEMPTS_TRY_LATER: 'Too many attempts. Please try again later',
  };
  return messages[code] || fallbackMessage;
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  // On mount, if we have a stored Firebase token, fetch user info.
  useEffect(() => {
    if (token) {
      axios
        .post(firebaseAuthUrl('accounts:lookup'), {
          idToken: token,
        })
        .then((res) => {
          const account = res.data?.users?.[0];
          if (!account) {
            throw new Error('User not found');
          }
          setUser({
            uid: account.localId,
            id: account.localId,
            localId: account.localId,
            name: account.displayName || account.email?.split('@')[0] || 'User',
            email: account.email,
          });
        })
        .catch(() => {
          // Token expired/invalid.
          localStorage.removeItem('token');
          setToken(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, [token]);

  const signup = async (name, email, password) => {
    try {
      const signUpRes = await axios.post(firebaseAuthUrl('accounts:signUp'), {
        email,
        password,
        returnSecureToken: true,
      });

      const updateRes = await axios.post(firebaseAuthUrl('accounts:update'), {
        idToken: signUpRes.data.idToken,
        displayName: name,
        returnSecureToken: true,
      });

      localStorage.setItem('token', updateRes.data.idToken);
      setToken(updateRes.data.idToken);
      setUser({
        uid: updateRes.data.localId || signUpRes.data.localId,
        id: updateRes.data.localId || signUpRes.data.localId,
        localId: updateRes.data.localId || signUpRes.data.localId,
        name: updateRes.data.displayName || name,
        email: updateRes.data.email || email,
      });
      return updateRes.data;
    } catch (error) {
      throw new Error(mapFirebaseError(error, 'Signup failed'));
    }
  };

  const login = async (email, password) => {
    try {
      const { data } = await axios.post(firebaseAuthUrl('accounts:signInWithPassword'), {
        email,
        password,
        returnSecureToken: true,
      });

      localStorage.setItem('token', data.idToken);
      setToken(data.idToken);
      setUser({
        uid: data.localId,
        id: data.localId,
        localId: data.localId,
        name: data.displayName || data.email?.split('@')[0] || 'User',
        email: data.email,
      });
      return data;
    } catch (error) {
      throw new Error(mapFirebaseError(error, 'Login failed'));
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, signup, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be inside AuthProvider');
  return ctx;
}
